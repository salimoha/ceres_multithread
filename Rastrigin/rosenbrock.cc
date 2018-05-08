// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/ceres.h"
#include "float.h"
#include "glog/logging.h"

typedef double(*FunctionSolver)(double*);

// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
class Rosenbrock : public ceres::FirstOrderFunction {
 public:
  virtual ~Rosenbrock() {}

  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const {
    const double x = parameters[0];
    const double y = parameters[1];

    cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
    if (gradient != NULL) {
      gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
      gradient[1] = 200.0 * (y - x * x);
    }
    return true;
  }

  virtual int NumParameters() const { return 2; }
};
omp_lock_t lock;
omp_lock_t lock_for_min_cost;
double rosenbrockSolver(double parameters[])
{
	double initialParameters[2] = { parameters[0],parameters[1] };
	ceres::GradientProblemSolver::Options options;
	options.minimizer_progress_to_stdout = false;

	ceres::GradientProblemSolver::Summary summary;
	ceres::GradientProblem problem(new Rosenbrock());
	ceres::Solve(options, problem, parameters, &summary);
	
	omp_set_lock(&lock);
	std::cout << summary.FullReport() << "\n";
	std::cout << "Initial x: " << initialParameters[0] << " y: " << initialParameters[1] << "\n";
	std::cout << "Final   x: " << parameters[0]
		<< " y: " << parameters[1] << "\n";
	omp_unset_lock(&lock);

	return summary.final_cost;
}
void minimizeFunction(FunctionSolver function, const int samples, double bestResult[])
{
	const double scale = 3.0;
	const double offset = 1.0;
	//std::vector<std::vector<double>> results(samples);
	double cost=DBL_MAX;
	srand(time(NULL));
	omp_init_lock(&lock);
	omp_init_lock(&lock_for_min_cost);
    #pragma omp parallel num_threads(samples)  
	{
		int i = omp_get_thread_num();
		double parameters[2];
		parameters[0] = (double)rand() / RAND_MAX; //value between 0 and 1;
		parameters[0] = parameters[0] * scale - offset;
		parameters[1] = (double)rand() / RAND_MAX; //value between 0 and 1;
		parameters[1] = parameters[1] * scale - offset;
		double costResult=function(parameters);
		omp_set_lock(&lock_for_min_cost);
		if (cost > costResult)
		{
			bestResult[0] = parameters[0];
			bestResult[1] = parameters[1];
			cost = costResult;
		}
		omp_unset_lock(&lock_for_min_cost);
	}
	omp_destroy_lock(&lock);	
	omp_destroy_lock(&lock_for_min_cost);
}
int main(int argc, char** argv) {
  int samples=5;
  if(argc==2)
	samples=atoi(argv[1]);
  google::InitGoogleLogging(argv[0]);

  double result[2];
  minimizeFunction(rosenbrockSolver, samples, result);
  std::cout << "best result is x:" << result[0] << " y:" << result[1] <<  std::endl;
  return 0;
}

