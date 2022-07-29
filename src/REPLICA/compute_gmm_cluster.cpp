#include "compute_gmm_cluster.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------- */
double Cluster::Gaussian(double val) {
  return _weight*(1/sqrt(2*M_PI*_sigma2))*exp(-0.5*(val-_mu)*(val-_mu)/_sigma2);
}

GMM_EM::GMM_EM(const std::vector<double>& points, int size, int iters, 
	double tolerate) :_points(points), _Csize(size), _iters(iters),
	_tolerate(tolerate) {
  _Psize = points.size();
  _clusters = KMeans_cluster(points);
  _old_log_likelihood = 0;
}

std::vector<Cluster> GMM_EM::KMeans_cluster(const std::vector<double>& points) {

  std::vector<std::vector<double> > point_set;
  for (int k = 0; k < _Csize; k++)
	point_set.push_back({});

  std::vector<double> means;
  std::vector<int> used_points;
  int pos_rand;
  for (int k = 0; k < _Csize; k++) {
	  while (true) {
	    pos_rand = rand() % _Psize;
	    if (std::find(used_points.begin(), used_points.end(), pos_rand)
			  == used_points.end()) {
		  means.emplace_back(points[pos_rand]);
		  used_points.push_back(pos_rand);
		  break;
	    }
	  }
  }

  bool done = false;
  double dist_min, temp, mean;
  std::vector<double>::iterator it;
  int pos_nearst, k;
  for (int iter = 0; iter < _iters; iter++) {
	for (k = 0; k < _Csize; k++)
	  point_set[k].clear();
	
	std::for_each(points.begin(), points.end(), [&](const double& p) {
		pos_nearst = 0;
		dist_min = abs(p - means[0]);
		for (k = 1; k < _Csize; k++) {
		  temp = abs(p - means[k]);
		  if (temp < dist_min) {
			dist_min = temp;
			pos_nearst = k;
			done = true;
		  }
		}
		point_set[pos_nearst].push_back(p);
	  });
	
	means.clear();
	for (k = 0; k < _Csize; k++) {
	  mean = 0;
	  for (std::vector<double>::iterator it = point_set[k].begin();
		  it != point_set[k].end(); ++it)
		mean += *it;
	  means.emplace_back(mean / point_set[k].size());
	}
	
	if (!done)
	  break;
  }

  std::vector<Cluster> clusters;
  for (int k = 0; k < _Csize; k++) {
	  Cluster ct;
	  ct._weight = static_cast<double>(point_set[k].size()) / _Psize;
	  ct._mu = mean = means[k];
	  double sigma_2 = 0;
	  std::for_each(point_set[k].begin(), point_set[k].end(), [&](double& p) {
	    sigma_2 += (p - mean) * (p - mean);
	    });
	  ct._sigma2 = sigma_2 / (_Psize - 1);
	  clusters.emplace_back(ct);
  }
  
  return clusters;
}

std::vector<std::vector<double> > GMM_EM::Expectation() {
  std::vector<std::vector<double> > gamma_hc;
  std::vector<double> gamma_c;
  double numerator, denominator;

  for (int p = 0; p < _Psize; p++) {	
	denominator = 0;
	gamma_c.clear();
	for (int c = 0; c < _Csize; c++) {
	  numerator = _clusters[c].Gaussian(_points[p]);
	  gamma_c.emplace_back(numerator);
	  denominator += numerator;
	}
	std::for_each(gamma_c.begin(), gamma_c.end(), 
	  [&](double& v) {v /= denominator; });
	gamma_hc.emplace_back(gamma_c);
  }
  
  return gamma_hc;
}

void GMM_EM::Maximization(std::vector<std::vector<double> >& gamma_pc) {

  for (int c = 0; c < _Csize; c++) {
	double Nc{0}, mu{0}, sigma2{0};
	for (int p = 0; p < _Psize; p++) {
	  Nc += gamma_pc[p][c];
	  mu += gamma_pc[p][c] * _points[p];
	}
	_clusters[c]._weight = Nc / _Psize;
	_clusters[c]._mu = mu = mu / Nc;

	for (int p = 0; p < _Psize; p++)
	  sigma2 += gamma_pc[p][c] * pow(_points[p] - mu, 2);
	_clusters[c]._sigma2 = sigma2 / Nc;
  }
}

void GMM_EM::run() {
  for (int iter = 0; iter < _iters; iter++) {
	std::vector<std::vector<double> > gamma = Expectation();
	Maximization(gamma);

  // log_likelihood function
	double log_likelihood = 0;
	for (int p = 0; p < _Psize; p++) {
	  double log_probability = 0;
	  for (int c = 0; c < _Csize; c++)
		log_probability += _clusters[c].Gaussian(_points[p]);
	  log_likelihood += log_probability;
	}

	double diff_log_likelihood = abs(log_likelihood - _old_log_likelihood);
	_old_log_likelihood = log_likelihood;

	if (diff_log_likelihood < _tolerate) break;
  }
}

void GMM_EM::output(double& mu, double& sigma2){
  int pos = 0;
  double temp_weight = _clusters[0]._weight;
  for (int c = 1; c < _Csize; c++)
    if (_clusters[c]._weight > temp_weight){
      temp_weight = _clusters[c]._weight;
      pos = c;
    }

  mu = _clusters[pos]._mu;
  sigma2 = _clusters[pos]._sigma2;
}
