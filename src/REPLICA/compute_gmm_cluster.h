#ifndef LMP_COMPUTE_GMM_CLUSTER_H
#define LMP_COMPUTE_GMM_CLUSTER_H

#include <vector>

namespace LAMMPS_NS{

struct Cluster {
  double _weight;
  double _mu, _sigma2;
  double Gaussian(double val);
};

class GMM_EM {
private:

  double _old_log_likelihood;
  std::vector<Cluster> _clusters;        
  std::vector<double> _points;

  int _Psize;
  int _Csize;		
  
  int _iters;		
  double _tolerate;

public:
  GMM_EM(const std::vector<double>&, int, int, double);
  
  // provide initial value for GMM
  std::vector<Cluster> KMeans_cluster(const std::vector<double>&);

  std::vector<std::vector<double> > Expectation();
  void Maximization(std::vector<std::vector<double> >&);
  void run();

  void output(double&, double&);
};

}   // namespace LAMMPS_NS

#endif
