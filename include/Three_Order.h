#ifndef THREE_ORDER_H
#define THREE_ORDER_H
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

class ThreeOrder {
private:
    int mx;  // Number of spatial grid points
    double CFL;
    double a, dx, dt;
    std::string folder;
    
    Eigen::VectorXd u_prev, u_new;

public:
    // Function prototypes
    double Initial_condition(double x);
    
    void Initialize(int mx_, double CFL_, double a, std::string folder_);
    void Inner();
    void Boundary();
    void Data(int i);
    void Forward();
    double dt_() { return dt; }
};

#endif