#ifndef UPWIND_H
#define UPWIND_H
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>

class WarmingBeam {
private:
    int mx;  // Number of spatial grid points
    long double CFL;
    long double a, dx, dt;
    std::string folder;

    // Use std::vector<long double> instead of Eigen::VectorXd for better compatibility with long double
    std::vector<long double> u_prev, u_new;

public:
    // Function prototypes
    long double Initial_condition(long double x);
    
    void Initialize(int mx_, long double CFL_, long double a_, std::string folder_);
    void Inner();
    void Boundary();
    void Data(int i);
    void Forward();
    long double dt_() { return dt; }
};
#endif