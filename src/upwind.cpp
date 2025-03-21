#include "upwind.h"

void UpWind::Initialize(int mx_, double CFL_, double a, std::string folder_) {
    mx = mx_;
    dx = 1.0 / mx;
    CFL = CFL_;
    dt = CFL * dx / a;

    u_prev.resize(mx + 1);
    u_new.resize(mx + 1);
    
    for (int i = 0; i <= mx; i++) {
        u_prev(i) = Initial_condition(i * dx - 0.5);
        u_new(i) = u_prev(i);
    }

    u_prev(0) = u_prev(mx);
    u_new(0) = u_new(mx);

    folder = folder_;
}

double UpWind::Initial_condition(double x) {
    if (x >= -0.5 && x < -0.25) return 0.0;
    else if (x >= -0.25 && x <= 0.25) return 1.0;
    else if (x > 0.25 && x <= 0.5) return 0.0;
    
    std::cout << "Error: Out of Range" << std::endl;
    return 0.0;
}


void UpWind::Boundary() {
    u_new(0) = u_prev(mx);   
}


void UpWind::Forward() {
    Boundary();
    for (int i = 1; i <= mx; i++) {
        u_new(i) = u_prev(i) - CFL * (u_prev(i) - u_prev(i - 1));
    }

    u_prev = u_new;
}


void UpWind::Data(int step) {
    std::ostringstream filename;
    filename << folder << "/output_step_" << step << ".txt";

    std::ofstream file(filename.str());
    for (int i = 0; i <= mx; i++) {
        file << i * dx << " " << u_new(i) << "\n";
    }
    file.close();
}
