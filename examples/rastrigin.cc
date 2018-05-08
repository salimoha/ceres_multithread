#include "ceres/ceres.h"
#include <iostream>
#include <fstream>
#include "float.h"
#include "omp.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

typedef double(*FunctionSolver)(double*, int);


omp_lock_t lock;
omp_lock_t lock_for_min_cost;

struct CostFunctor1D {
	template <typename T> bool operator()(const T* const x1, T* residual) const {
		residual[0] = T(10.0) + x1[0] * x1[0] - T(10.0)*cos(T(2) * T(M_PI)*x1[0]);
		return true;
	}
};

struct CostFunctor2D {
	template <typename T> bool operator()(const T* const x1, const T* const x2, T* residual) const {
		residual[0] = T(20.0) + x1[0] * x1[0] - T(10.0)*cos(T(2 * M_PI)*x1[0])
			+ x2[0] * x2[0] - T(10.0)*cos(T(2 * M_PI)*x2[0]);
		return true;
	}
};

struct CostFunctor3D {
	template <typename T> bool operator()(const T* const x1, const T* const x2, const T* const x3, T* residual) const {
		residual[0] = T(30.0) + x1[0] * x1[0] - T(10.0)*cos(T(2 * M_PI)*x1[0])
			+ x2[0] * x2[0] - T(10.0)*cos(T(2 * M_PI)*x2[0])
			+ x3[0] * x3[0] - T(10.0)*cos(T(2 * M_PI)*x3[0]);
		return true;
	}
};

struct CostFunctor6D {
	template <typename T> bool operator()(const T* const x1, 
		const T* const x2,
		const T* const x3,
		const T* const x4,
		const T* const x5,
		const T* const x6,
		T* residual) const {
		residual[0] = T(60.0) + x1[0] * x1[0] - T(10.0)*cos(T(2 * M_PI)*x1[0])			
			+ x2[0] * x2[0] - T(10.0)*cos(T(2 * M_PI)*x2[0])
			+ x3[0] * x3[0] - T(10.0)*cos(T(2 * M_PI)*x3[0])
			+ x4[0] * x4[0] - T(10.0)*cos(T(2 * M_PI)*x4[0])
			+ x5[0] * x5[0] - T(10.0)*cos(T(2 * M_PI)*x5[0])
			+ x6[0] * x6[0] - T(10.0)*cos(T(2 * M_PI)*x6[0]);
		return true;
	}
};

/*struct CostFunctorND {
	template <typename T> bool operator()(const T* const x,		
		const T* const n,
		T* residual) const {
		residual[0] = 10.0*n[0];
		for (auto index = 0; index < n[0]; index++)
		{
			residual[0] += x[index] * x[index] - 10.0*cos(2 * M_PI*x[index]);
		}
		return true;
	}
};*/

double SolveRastrigin(double parameters[], int dimension)
{
	double* initialParameters = new double[dimension];
	for (int i = 0; i < dimension; i++)
		initialParameters[i] = parameters[i];
	Problem problem;

	// Set up the only cost function (also known as residual). This uses
	// auto-differentiation to obtain the derivative (jacobian).
	CostFunction* cost_function;
	if (dimension == 1)
	{
		cost_function = new AutoDiffCostFunction<CostFunctor1D, 1, 1>(new CostFunctor1D);
		problem.AddResidualBlock(cost_function, NULL, &parameters[0]);
	}
	if (dimension == 2)
	{
		cost_function = new AutoDiffCostFunction<CostFunctor2D, 1, 1, 1>(new CostFunctor2D);
		problem.AddResidualBlock(cost_function, NULL, &parameters[0],
			&parameters[1]);
	}
	if (dimension == 3)
	{
		cost_function = new AutoDiffCostFunction<CostFunctor3D, 1, 1, 1, 1>(new CostFunctor3D);
		problem.AddResidualBlock(cost_function, NULL, &parameters[0],
			&parameters[1], &parameters[2]);
	}
	if (dimension == 6)
	{
		cost_function = new AutoDiffCostFunction<CostFunctor6D, 1, 1, 1, 1, 1, 1, 1>(new CostFunctor6D);
		problem.AddResidualBlock(cost_function, NULL,
			&parameters[0], &parameters[1], &parameters[2],
			&parameters[3], &parameters[4], &parameters[5]);
	}
	//cost_function = new AutoDiffCostFunction<CostFunctorND, 1, dimension,1>(new CostFunctorND);
	//problem.AddResidualBlock(cost_function, NULL, x);

	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	Solver::Summary summary;
	Solve(options, &problem, &summary);

	omp_set_lock(&lock);
	std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial x:(";
	for (int i = 0; i < dimension; i++)
	{
		std::cout << "  " << initialParameters[i] << "  ";
	}
	std::cout << ")." << std::endl;
	std::cout << "Final   x:(";
	for (int i = 0; i < dimension; i++)
	{
		std::cout << "  " << parameters[i] << "  ";
	}
	std::cout << ")." << std::endl;
	omp_unset_lock(&lock);
	delete[] initialParameters;
	return summary.final_cost;
}




void minimizeFunction(FunctionSolver function,int dimension,int samples, double bestResult[])
{
	const double scale = 3.0;
	const double offset = 1.0;
	//std::vector<std::vector<double>> results(samples);
	double cost = DBL_MAX;
	srand(time(NULL));
	std::vector<double> random_values(samples * dimension);
	for (int i = 0; i < samples * dimension; i++)
	{
		random_values[i] = ((double)rand() / RAND_MAX)*scale - offset;
	}
	omp_init_lock(&lock);
	omp_init_lock(&lock_for_min_cost);
	omp_set_num_threads(samples);
	#pragma omp parallel  
	{
		int th = omp_get_thread_num();
		double *parameters = new double[dimension];
		for (int i = 0; i < dimension; i++)
		{
			parameters[i] = random_values[th*dimension + i]; //value between 0 and 1;			
		}
		
		double costResult = function(parameters,dimension);
		omp_set_lock(&lock_for_min_cost);
		if (cost > costResult)
		{
			for (int i = 0; i < dimension; i++)
			{
				bestResult[i]=parameters[i] ; 				
			}			
			cost = costResult;
		}
		omp_unset_lock(&lock_for_min_cost);
		delete[] parameters;
	}
	omp_destroy_lock(&lock);
	omp_destroy_lock(&lock_for_min_cost);
}
int main(int argc, char** argv) {
	google::InitGoogleLogging(argv[0]);
	int dimension = 1;
	if (argc >= 2)
		dimension = atoi(argv[1]);
	int openMP_Threads = 1;
	if (argc == 3)
		openMP_Threads = atoi(argv[2]);
	double *result=new double[dimension];
	minimizeFunction(SolveRastrigin, dimension, openMP_Threads,result);
	std::cout << "best result is x:(";
	for (int i = 0; i < dimension; i++)
	{
		std::cout << "  " << result[i] << "  ";
	}
	std::cout << ")." << std::endl;
	delete[] result;
	return 0;
}

