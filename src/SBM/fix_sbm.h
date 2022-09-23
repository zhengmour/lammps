/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(shake,FixSBM);
// clang-format on
#else

#ifndef LMP_FIX_SBM_H
#define LMP_FIX_SBM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSBM : public Fix {
 public:
  FixSBM(class LAMMPS *, int, char **);
  ~FixSBM() override;

  int setmask() override;

  void init() override;
  void setup(int) override;
  void pre_neighbor() override;
  void post_force(int) override;

  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  void update_arrays(int, int) override;
  void set_molecule(int, tagint, int, double *, double *, double *) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  virtual void shake_end_of_step(int vflag);
  virtual void correct_coordinates(int vflag);
  virtual void correct_velocities();

  int dof(int) override;
  void reset_dt() override;
  void *extract(const char *, int &) override;

 protected:

  int me, nprocs;
  bigint next_output;

  // settings from input command
  // native term, excluded volume term and non-native term(optional)
  int *native_flag;             // native term to constrain
  int *native_type_flag;        // constrain native terms to these types

  int *excvol_flag;             // excluded volume term to constrain
  int *excvol_type_flag;        // constrain excluded terms to these types

  int *nonative_flag;           // non-native term to constrain
  int *nonative_type_flag;      // constrain non-native terms to these types

  double **x, **v, **f;         // local ptrs to atom class quantities
  double **ftmp, **vtmp;        // pointers to temporary arrays for f,v

  double *mass, *rmass;
  int *type;
  int nlocal;

  // pair coefficients of each term
  // coefficients are not assigned by type, but by each term
  double cut_global;
  double **cut;
  double **native_epsilon, **native_sigma;
  double **excvol_epsilon, **excvol_sigma;
  double **nonative_epsilon, **nonative_sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;

};

}    // namespace LAMMPS_NS

#endif
#endif
