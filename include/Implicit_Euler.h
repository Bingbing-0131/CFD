#ifndef IMPLICITEULER_H
#define IMPLICITEULER_H

#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <sstream>

class ImplicitEuler {
private:
    int mx;
    double CFL;
    double a, dx, dt;
    std::string folder;

    Eigen::VectorXd u_prev, u_new;
    Eigen::MatrixXd A;

public:
    double Initial_condition(double x);
    void Initialize(int mx_, double CFL_, double a_, std::string folder_);
    void BuildMatrix();
    void Forward();
    void Data(double time);
    double dt_() { return dt; }
};

#endif
