/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: zhengda
   temper/alter N M L temp fix-ID ctol maxiter seed1 seed2 index
------------------------------------------------------------------------- */

#include "temper_alter.h"
#include "compute_gmm_cluster.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "random_park.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "memory.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdlib.h>

using namespace LAMMPS_NS;

//#define TEMPER_ALTER_DEBUG 1
//#define TEMPER_ALTER_TEMP_DEBUG 1 

/* ---------------------------------------------------------------------- */

TemperAlter::TemperAlter(LAMMPS *lmp) : Command(lmp) {}

/* ---------------------------------------------------------------------- */

TemperAlter::~TemperAlter()
{
	MPI_Comm_free(&roots);
	if (ranswap) delete ranboltz;
	delete ranboltz;
	delete [] set_temp;
	delete [] temp2world;
	delete [] world2temp;
	delete [] world2root;

  delete [] set_sigma2;
  delete [] set_mu;
  delete [] set_pe;
}

/* ----------------------------------------------------------------------
   perform tempering with inter-world swaps
------------------------------------------------------------------------- */

void TemperAlter::command(int narg, char **arg)
{
  if (universe->nworlds == 1)
    error->all(FLERR,"Must have more than one processor partition to temper");
  if (domain->box_exist == 0)
    error->all(FLERR,"Temper command before simulation box is defined");
  if (narg != 9 && narg != 10)
    error->universe_all(FLERR,"Illegal temper command");

  int nsteps = utils::inumeric(FLERR,arg[0],false,lmp);
  nevery = utils::inumeric(FLERR,arg[1],false,lmp);
  nalter = utils::inumeric(FLERR,arg[2],false,lmp);
  double temp = utils::numeric(FLERR,arg[3],false,lmp);
  ctol = utils::numeric(FLERR, arg[5],false,lmp);
  maxiter = utils::numeric(FLERR, arg[6],false,lmp);

  if (ctol < 0.0 || maxiter < 0)
     error->all(FLERR,"Illegal temper/alter command");

  // ignore temper command, if walltime limit was already reached
  if (timer->is_timeout()) return;

  // Get and check if temp fix exists
  for (whichfix = 0; whichfix < modify->nfix; whichfix++)
	if (strcmp(arg[4],modify->fix[whichfix]->id) == 0) break;
  if (whichfix == modify->nfix)
    error->universe_all(FLERR,"Tempering fix ID is not defined");
  
  seed_swap = utils::inumeric(FLERR,arg[7],false,lmp);
  seed_boltz = utils::inumeric(FLERR,arg[8],false,lmp);

  my_set_temp = universe->iworld;
  if (narg==10) my_set_temp = utils::inumeric(FLERR,arg[9],false,lmp);
  if ((my_set_temp < 0) || (my_set_temp >= universe->nworlds))
  	error->universe_one(FLERR,"Illegal temperature index");

  // swap frequency must evenly divide total # of timesteps
  if (nevery <= 0)
    error->universe_all(FLERR,"Invalid frequency in temper command");
  nswaps = nsteps/nevery;
  if (nswaps*nevery != nsteps)
    error->universe_all(FLERR,"Non integer # of swaps in temper command");

  // fix style must be appropriate for temperature control, i.e. it needs
  // to provide a working Fix::reset_target() and must not change the volume.

  if ((!utils::strmatch(modify->fix[whichfix]->style,"^nvt")) &&
      (!utils::strmatch(modify->fix[whichfix]->style,"^langevin")) &&
      (!utils::strmatch(modify->fix[whichfix]->style,"^gl[de]$")) &&
      (!utils::strmatch(modify->fix[whichfix]->style,"^rigid/nvt")) &&
      (!utils::strmatch(modify->fix[whichfix]->style,"^temp/")))
    error->universe_all(FLERR,"Tempering temperature fix is not supported");

  // setup for long tempering run

  update->whichflag = 1;
  timer->init_timeout();

  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");

  lmp->init();

  // local storage

  me_universe = universe->me;
  MPI_Comm_rank(world,&me);
  nworlds = universe->nworlds;
  iworld = universe->iworld;
  boltz = force->boltz;

  // pe_compute = ptr to thermo_pe compute
  // notify compute it will be called at first swap
  
  int step_pe = (int)std::max(1.0, nevery*nalter*0.001);
  int id = modify->find_compute("thermo_pe");
  if (id < 0) error->all(FLERR,"Tempering could not find thermo_pe compute");
  Compute *pe_compute = modify->compute[id];
  pe_compute->addstep(update->ntimestep + step_pe);

  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(universe->uworld,color,0,&roots);

  // RNGs for swaps and Boltzmann test
  // warm up Boltzmann RNG

  if (seed_swap) ranswap = new RanPark(lmp,seed_swap);
  else ranswap = nullptr;
  ranboltz = new RanPark(lmp,seed_boltz + me_universe);
  for (int i = 0; i < 100; i++) ranboltz->uniform();

  // world2root[i] = global proc that is root proc of world i

  world2root = new int[nworlds];
  if (me == 0)
    MPI_Allgather(&me_universe,1,MPI_INT,world2root,1,MPI_INT,roots);
  MPI_Bcast(world2root,nworlds,MPI_INT,0,world);

  // create static list of set temperatures
  // allgather tempering arg "temp" across root procs
  // bcast from each root to other procs in world
  
  set_temp = new double[nworlds];
  if (me == 0) MPI_Allgather(&temp,1,MPI_DOUBLE,set_temp,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);

  // create world2temp only on root procs from my_set_temp
  // create temp2world on root procs from world2temp,
  //   then bcast to all procs within world

  world2temp = new int[nworlds];
  temp2world = new int[nworlds];
  if (me == 0) {
    MPI_Allgather(&my_set_temp,1,MPI_INT,world2temp,1,MPI_INT,roots);
    for (int i = 0; i < nworlds; i++) temp2world[world2temp[i]] = i;
  }
  MPI_Bcast(temp2world,nworlds,MPI_INT,0,world);

  // create set_sigma2 stores sigma by fitting for each process
  // create set_mu stores mean by fitting for each process
  // create set_pe stores pe between swap
  set_sigma2 = new double[nworlds];
  set_mu = new double[nworlds];
  int set_pe_len=(int)(nevery*nalter/step_pe);
  set_pe = new double[set_pe_len];

  // if restarting tempering, reset temp target of Fix to current my_set_temp
  if (narg == 9) {
    double new_temp = set_temp[my_set_temp];
    modify->fix[whichfix]->reset_target(new_temp);
  }

  // setup tempering runs

  int i,which,partner,swap,partner_set_temp,partner_world;
  double pe,pe_partner,boltz_factor,new_temp;
  
  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen, "Setting up tempering...\n");

  update->integrate->setup(1);

  if (me_universe == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->uscreen," T%d",i);
      fprintf(universe->uscreen,"\n");
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->ulogfile," T%d",i);
      fprintf(universe->ulogfile,"\n");
    }
    print_status();
  }

  timer->init();
  timer->barrier_start();
  
  int istep_mod = 0, start = 0;
  int shift = 0;
  for (int iswap = 0; iswap < nswaps; iswap++) {
    // run for nevery timesteps
    // store enegy for each step
    timer->init_timeout();
    
    if (istep_mod % set_pe_len == 0)
      istep_mod = 0;
    start = istep_mod;
    
    int istep;
    for (istep=-shift; istep<nevery; istep+=step_pe){
      update->integrate->run(step_pe);
      
	    pe = pe_compute->compute_scalar();
      pe_compute->addstep(update->ntimestep + step_pe);
      set_pe[istep_mod] = pe;
      
      istep_mod += 1;
    }
    shift = nevery - istep; 
    
    // swap pe between different procs
    int len = istep_mod - start;
    double* set_pe_tmp = new double[len];
    MPI_Send(set_pe+start,len,MPI_DOUBLE,world2temp[me_universe],0,universe->uworld);
    MPI_Recv(set_pe_tmp,len,MPI_DOUBLE,temp2world[me_universe],0,universe->uworld,MPI_STATUS_IGNORE);
    MPI_Barrier(universe->uworld);

    int my_timeout=0;
    int any_timeout=0;
    if (timer->is_timeout()) my_timeout=1;
    MPI_Allreduce(&my_timeout, &any_timeout, 1, MPI_INT, MPI_SUM, universe->uworld);
    if (any_timeout) {
      timer->force_timeout();
      break;
    }

    // compute PE
    // notify compute it will be called at next swap

    // pe = pe_compute->compute_scalar();
    // pe_compute->addstep(update->ntimestep + nevery);

    // which = which of 2 kinds of swaps to do (0,1)

    if (!ranswap) which = iswap % 2;
    else if (ranswap->uniform() < 0.5) which = 0;
    else which = 1;

    // partner_set_temp = which set temp I am partnering with for this swap

    if (which == 0) {
      if (my_set_temp % 2 == 0) partner_set_temp = my_set_temp + 1;
      else partner_set_temp = my_set_temp - 1;
    } else {
      if (my_set_temp % 2 == 1) partner_set_temp = my_set_temp + 1;
      else partner_set_temp = my_set_temp - 1;
    }

    // partner = proc ID to swap with
    // if partner = -1, then I am not a proc that swaps

    partner = -1;
    if (me == 0 && partner_set_temp >= 0 && partner_set_temp < nworlds) {
      partner_world = temp2world[partner_set_temp];
      partner = world2root[partner_world];
    }

    // swap with a partner, only root procs in each world participate
    // hi proc sends PE to low proc
    // lo proc make Boltzmann decision on whether to swap
    // lo proc communicates decision back to hi proc

    swap = 0;
    if (partner != -1) {
      if (me_universe > partner)
        MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
      else
        MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);

      if (me_universe < partner) {
        boltz_factor = (pe - pe_partner) *
          (1.0/(boltz*set_temp[my_set_temp]) -
           1.0/(boltz*set_temp[partner_set_temp]));
        if (boltz_factor >= 0.0) swap = 1;
        else if (ranboltz->uniform() < exp(boltz_factor)) swap = 1;
      }

      if (me_universe < partner)
        MPI_Send(&swap,1,MPI_INT,partner,0,universe->uworld);
      else
        MPI_Recv(&swap,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);

#ifdef TEMPER_ALTER_DEBUG
      if (me_universe < partner)
        printf("SWAP %d & %d: yes = %d,Ts = %d %d, PEs = %g %g, Bz = %g %g\n",
               me_universe,partner,swap,my_set_temp,partner_set_temp,
               pe,pe_partner,boltz_factor,exp(boltz_factor));
#endif

    }

    // bcast swap result to other procs in my world

    MPI_Bcast(&swap,1,MPI_INT,0,world);

    // rescale kinetic energy via velocities if move is accepted

    if (swap) scale_velocities(partner_set_temp,my_set_temp);

    // if my world swapped, all procs in world reset temp target of Fix

    if (swap) {
      new_temp = set_temp[partner_set_temp];
      modify->fix[whichfix]->reset_target(new_temp);
    }

    // update my_set_temp and temp2world on every proc
    // root procs update their value if swap took place
    // allgather across root procs
    // bcast within my world

    if (swap) my_set_temp = partner_set_temp;
    if (me == 0) {
      MPI_Allgather(&my_set_temp,1,MPI_INT,world2temp,1,MPI_INT,roots);
      for (i = 0; i < nworlds; i++) temp2world[world2temp[i]] = i;
    }
    MPI_Bcast(temp2world,nworlds,MPI_INT,0,world);
   
    double sigma2, mu;
    // recalculate the temperature sequence every nalter swaps
    if ((iswap+1) % nalter == 0 && iswap != (nswaps - 1)){
      
      // using gmm cluster fitting distribution
      fit_distribution(mu, sigma2, set_pe_len); 

      if (me == 0){
        MPI_Allgather(&sigma2,1,MPI_DOUBLE,set_sigma2,1,MPI_DOUBLE,roots);
        MPI_Allgather(&mu,1,MPI_DOUBLE,set_mu,1,MPI_DOUBLE,roots);
      }
      MPI_Bcast(set_sigma2,nworlds,MPI_DOUBLE,0,world);
      MPI_Bcast(set_mu,nworlds,MPI_DOUBLE,0,world);
      
      MPI_Barrier(universe->uworld);
      if (me == 0) generate_temp_squence();
      MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);
      

      // nvt langevin gl[de] rigid/nvt temp/     
      double new_temp = set_temp[world2temp[me_universe]];
      modify->fix[whichfix]->reset_target(new_temp);

      if (me_universe == 0) print_temperature();
    }
    // print out current swap status

    if (me_universe == 0) print_status();
  }

  timer->barrier_stop();

  update->integrate->cleanup();

  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}


/* ----------------------------------------------------------------------
   scale kinetic energy via velocities a la Sugita
------------------------------------------------------------------------- */
void TemperAlter::scale_velocities(int t_partner, int t_me)
{
  double sfactor = sqrt(set_temp[t_partner]/set_temp[t_me]);

  double **v = atom->v;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    v[i][0] = v[i][0]*sfactor;
    v[i][1] = v[i][1]*sfactor;
    v[i][2] = v[i][2]*sfactor;
  }
}

/* ----------------------------------------------------------------------
   proc 0 prints current tempering status
------------------------------------------------------------------------- */
void TemperAlter::print_status()
{
  if (universe->uscreen) {
    fprintf(universe->uscreen,BIGINT_FORMAT,update->ntimestep);
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen," %d",world2temp[i]);
    fprintf(universe->uscreen,"\n");
  }
  if (universe->ulogfile) {
    fprintf(universe->ulogfile,BIGINT_FORMAT,update->ntimestep);
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->ulogfile," %d",world2temp[i]);
    fprintf(universe->ulogfile,"\n");
    fflush(universe->ulogfile);
  }
}

void TemperAlter::print_temperature()
{
  if (universe->uscreen) {
    fprintf(universe->uscreen,"Temp");
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen," %.4f",set_temp[i]);
    fprintf(universe->uscreen,"\n");
  }
}


/* ----------------------------------------------------------------------
   fit the energy distribution
------------------------------------------------------------------------- */
void TemperAlter::fit_distribution(double& mu, double& sigma_2, int arr_len)
{ 
  int nbins = arr_len * 0.05;
  int size = 3;
  std::vector<double> points(set_pe, set_pe + arr_len);

  // using GMM to fitting datas, and extracting 
  // the gaussian distribution with the largest proportion
  GMM_EM gmm = GMM_EM(points, size, maxiter, ctol);
  gmm.run();
  gmm.output(mu, sigma_2);
}


/* ----------------------------------------------------------------------
   calculate complementary error function
------------------------------------------------------------------------- */
double TemperAlter::erfc(double x) {
  // constants
  double a1 = 0.254829592;
  double a2 = -0.284496736;
  double a3 = 1.421413741;
  double a4 = -1.453152027;
  double a5 = 1.061405429;
  double p = 0.3275911;

  // Save the sign of x
  int sign = 1;
  if (x < 0) {
    sign = -1;
  }
  x = abs(x);

  // Abramowitz, M.and Stegun, I.A. 1972.
      // Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables.Dover.
      // Formula 7.1.26
  double t = 1.0 / (1.0 + p * x);
  double y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

  return sign * y;
}

/* ----------------------------------------------------------------------
   calculate the second integral term of p(T1,T2)
------------------------------------------------------------------------- */
double TemperAlter::integral2(double mu12, double sig12, double CC) {
  double integ = 0.;
  double umax = mu12 + 5 * sig12;
  double delta_u = umax * 0.001;

  for (double u = 0; u < umax; u += delta_u) {
    double modify_u = u + 0.5 * delta_u;
    double argument = - CC * modify_u - (modify_u - mu12) * (modify_u - mu12) / (2 * sig12 * sig12);
    integ += exp(argument);
  }
  
  return delta_u * integ / (sig12 * sqrt(2 * M_PI));
}

/* ----------------------------------------------------------------------
   probability P which depends on the reference temperatures of 
   two replicas (T1, T2) and instantaneous potential energies (U1, U2)
   P = 1/2[1+erf(-mu/sigma��2)] + 1/2e^(CB)[1+erf(B/sigma��2)]
------------------------------------------------------------------------- */
double TemperAlter::probability(int n1, int n2)
{
  double cc, sig12_2, mu12;
  cc =  1/boltz * (1/set_temp[n1]-1/set_temp[n2]);
  sig12_2 = set_sigma2[n1] + set_sigma2[n2];
  mu12 = set_mu[n2] - set_mu[n2];

  // the first term of integral
  double erfarg1 = mu12 / sqrt(sig12_2 * 2);
  double Integ1 = 0.5 * (erfc(erfarg1));

  // the second term of integral
  // Old analytical code according to the paper, however
  // this suffers from numerical issues in extreme cases.
  // exparg  = CC*(-mu12 + CC*var/2);
  // erfarg2 = (mu12 - CC*var)/(sig12*sqrt(2));
  // I2      = 0.5*exp(exparg)*(1.0 + erf(erfarg2));
  // Use numerical integration instead.
  double Integ2 = integral2(mu12, sqrt(sig12_2), cc);
  
  return (Integ1+Integ2);
}
double TemperAlter::probability(int n1, int n2, double T1, double T2)
{
  double cc, sig12_2, mu12;
  cc =  1/boltz * (1/T1-1/T2);
  sig12_2 = set_sigma2[n1] + set_sigma2[n2];
  mu12 = set_mu[n2] - set_mu[n2];

  // the first term of integral
  double erfarg1 = mu12 / sqrt(sig12_2 * 2);
  double Integ1 = 0.5 * (erfc(erfarg1));

  // the second term of integral
  // Old analytical code according to the paper, however
  // this suffers from numerical issues in extreme cases.
  // exparg  = CC*(-mu12 + CC*var/2);
  // erfarg2 = (mu12 - CC*var)/(sig12*sqrt(2));
  // I2      = 0.5*exp(exparg)*(1.0 + erf(erfarg2));
  // Use numerical integration instead.
  double Integ2 = integral2(mu12, sqrt(sig12_2), cc);
  
  return (Integ1+Integ2);
}

/* ----------------------------------------------------------------------
   the ratio of probability and first-order derivative
------------------------------------------------------------------------- */
double TemperAlter::deriv_prob(int n1, int n2, bool flag)
{
  double cc, sig12_2, mu12;
  double inv_T1, inv_T2;
  cc =  1/boltz * (1/set_temp[n1]-1/set_temp[n2]);
  sig12_2 = set_sigma2[n1] + set_sigma2[n2];
  mu12 = set_mu[n2] - set_mu[n2];
  inv_T1 = 1 / set_temp[n1];
  inv_T2 = 1 / set_temp[n2];

  // the second term of probability
  double Integ2 = integral2(mu12, sqrt(sig12_2), cc);

  // the first term of first-order derivative
  double pre_expfrag = mu12 + cc * sig12_2;
  double Differ1 = pre_expfrag * Integ2;

  // the second term of first-order derivative
  double exparg2 = cc * (mu12 + 0.5 * cc * sig12_2);
  double erfarg2 = (mu12 + cc * sig12_2) / sqrt(sig12_2 * 2);
  double Differ2 = sqrt(0.5*sig12_2/M_PI) * exp(exparg2 - erfarg2 * erfarg2);

  double multifrag;
  if (flag)
    multifrag =  1/boltz * (inv_T1 + inv_T2 * inv_T2);
  else 
    multifrag =  - 1/boltz * (inv_T1 * inv_T1 + inv_T2);  

  return multifrag * (Differ1+Differ2);
}


/* ----------------------------------------------------------------------
   generate the dependent variable vector of p(T1,T2) - p(T12,T3)
------------------------------------------------------------------------- */
vector<double> TemperAlter::generate_delta_prob(int size) {
  vector<double> probability_diff;
  
  for (int i = 0; i < size; ++i) {
    double diff = probability(i+1, i+2) - probability(i, i+1);
    probability_diff.emplace_back(diff);
  }

  return probability_diff;
}

/* ----------------------------------------------------------------------
   generate the Jacobian of the p(T1,T2) - p(T12,T3)
------------------------------------------------------------------------- */
vector<vector<double> > TemperAlter::generate_jacobian(int size) {
  vector<vector<double> > jacobian;
  // initialize jacobian matrix
  for (int i = 0; i < size; ++i){
    vector<double> v(size);
    jacobian.emplace_back(v);
  }

  for (int i = 0; i < size; ++i) {
    if (i == 0) {
      jacobian[i][i] = deriv_prob(i, i+1, 1) - deriv_prob(i+1, i+2, 0);
      jacobian[i][i+1] = - deriv_prob(i+1, i+2, 1);
    }
    else if (i == size - 1) {
      jacobian[i][i-1] = deriv_prob(i-1, i, 0);
      jacobian[i][i] = deriv_prob(i, i+1, 1) - deriv_prob(i+1, i+2, 0);
    }
    else {
      jacobian[i][i-1] = deriv_prob(i-1, i, 0);
      jacobian[i][i] = deriv_prob(i, i+1, 1) - deriv_prob(i+1, i+2, 0);
      jacobian[i][i+1] = - deriv_prob(i+1, i+2, 1);
    }
  }

  return jacobian;
}

/* ----------------------------------------------------------------------
   calculate (s2,s3,s4..) from Js = -F
------------------------------------------------------------------------- */
void TemperAlter::calc_equation_set(vector<vector<double> > jacobian, 
        vector<double> probability_diff, int size) {
  double factor = 0;
  double eps = 1e-5;
  vector<double> delta_temp(size);
    
  for (int j = 0; j < size - 1; j++) {
    if (abs(jacobian[j][j]) < eps) 
      error->all(FLERR,"zero pivot encountered");
    for (int i = j + 1; i < size; i++) {
      factor = jacobian[i][j] / jacobian[j][j];
      for (int k = j + 1; k < size; k++)
        jacobian[i][k] -= factor * jacobian[j][k];
      probability_diff[i] += factor * probability_diff[j];     
    }
  }

  for (int i = size - 1; i >= 0; i--) {
    for (int j = i + 1; j < size; j++)
      probability_diff[i] -= jacobian[i][j] * delta_temp[j];
    delta_temp[i] = probability_diff[i] / jacobian[i][i];
  }

  for (int i = 0; i < size; i++)
    set_temp[i+1] += delta_temp[i];

  // generate new temperature
  for (int i = 0; i < size; i++) {
    delta_temp[i] = 0;
    for (int j = 0; j < size; j++)
      delta_temp[i] += jacobian[i][j] * probability_diff[j];
    set_temp[i] += delta_temp[i]; 
  }
}

/* ----------------------------------------------------------------------
   generate temperature squence 
------------------------------------------------------------------------- */
void TemperAlter::generate_temp_squence(int size)
{
  double boltz = force->boltz;
  int iter = 0;
  double max_diff = 0;

  do {
    vector<double> old_set_temp(set_temp, set_temp + nworlds);

    vector<double> probability_diff = generate_delta_prob(size);
    vector<vector<double> > jacobian = generate_jacobian(size);
    calc_equation_set(jacobian, probability_diff, size);

    // condition of convergence
    max_diff = 0;
    for(int i = 0; i < nworlds; i++)
      if(abs(old_set_temp[i] - set_temp[i]) > max_diff)
        max_diff = abs(old_set_temp[i] - set_temp[i]); 
      
    iter += 1;
  } while (max_diff > ctol && iter < maxiter);
}

void TemperAlter::generate_temp_squence() {

#ifdef TEMPER_ALTER_TEMP_DEBUG
  if (universe->uscreen){ 
    fprintf(universe->uscreen, "average ");
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen, " %.4f", set_mu[i]);
    fprintf(universe->uscreen, "\nsigma^2 ");
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen, " %.4f", set_sigma2[i]);
    fprintf(universe->uscreen, "\ntemper ");
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen, " %.4f", set_temp[i]);
    fprintf(universe->uscreen, "\n");
  }
#endif

  double piter = probability(0, 1);
  double ratio = piter;
  double Tmax = set_temp[nworlds - 1];
  int p_forward;

  double p_low = 0., p_high=1.;

  int loop = 0;
  do {  
    for (int n = 0; n < nworlds - 1; n++) {
      int forward = 1, iter = 0;

      double T1 = set_temp[n];
      double T2 = T1 + 1;

      double low = T1;
      double high = T2; 

     do {
        piter = probability(n, n + 1, T1, T2);
        if (piter > ratio) {
          if (forward == 1) T2 += 1.;
          else if (forward == 0) {
            low = T2;
            T2 = low + (high - low) * 0.5;
          }
        }
        else if (piter < ratio) {
          if (forward == 1) {
            forward = 0;
            low = T2 - 1.;
          }
          else if (forward == 0) {
            high = T2;
            T2 = low + (high - low) * 0.5;
          }
        }
      
        iter++;
      } while(abs(piter-ratio) > ctol && iter < maxiter); 

      set_temp[n + 1] = T2;
    }
    
    if (loop == 0)
      p_forward = (set_temp[nworlds-1] - Tmax) ? 1 : 0;
    
    if (set_temp[nworlds - 1] > Tmax) {
      if (p_forward == 1) ratio += 0.01;
      else if (p_forward == 0) {
        p_low = ratio;
        ratio = p_low + (p_high - p_low) * 0.5;
      }
      if (ratio > 1.0)
        ratio = 1.0;
    }
    else if (set_temp[nworlds - 1] < Tmax) {
      if (p_forward == 1) {
        p_forward = 0;
        p_low = ratio - 0.01;
      }
      else if (p_forward == 0) {
        p_high = ratio;
        ratio = p_low + (p_high - p_low) * 0.5;
      }
    }
    
    loop += 1;
  } while (abs(set_temp[nworlds - 1] - Tmax) > 1000 * ctol && loop < maxiter);
   
  set_temp[nworlds - 1] = Tmax;
}
