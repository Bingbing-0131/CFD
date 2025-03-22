#include "Implicit_Euler.h"

void ImplicitEuler::Initialize(int mx_, double CFL_, double a_, std::string folder_) {
    mx = mx_;
    dx = 1.0 / mx;
    CFL = CFL_;
    a = a_;
    dt = CFL * dx / a;

    u_prev.resize(mx + 1);
    u_new.resize(mx + 1);
    A.resize(mx + 1, mx + 1);

    for (int i = 0; i <= mx; i++) {
        u_prev(i) = Initial_condition(i * dx - 0.5);
        u_new(i) = u_prev(i);
    }

    folder = folder_;
    BuildMatrix();  
}

double ImplicitEuler::Initial_condition(double x) {
    if (x >= -0.5 && x < -0.25) return 0.0;
    else if (x >= -0.25 && x <= 0.25) return 1.0;
    else if (x > 0.25 && x <= 0.5) return 0.0;

    std::cout << "Error: Out of Range" << std::endl;
    return 0.0;
}

void ImplicitEuler::BuildMatrix() {
    A.setZero();
    for (int i = 0; i < = mx; ++i) {
        A(i, i) = 1.0;

        // Central difference (symmetric)
        int ip = (i + 1) ;           // i+1 with periodic wrap
        int im = (i - 1 ); // i-1 with periodic wrap

        if(ip>(mx))
        {
            ip = 0;
        }
        if(im<0)
        {
            im = mx;
        }

        A(i, ip) = CFL / 2.0;
        A(i, im) = -CFL / 2.0;
    }
}


void ImplicitEuler::Forward() {
    u_new = A.colPivHouseholderQr().solve(u_prev);

    u_prev = u_new;
}

void ImplicitEuler::Data(double time) {
    std::ostringstream filename;
    filename << folder << "/implicit_" << std::fixed << std::setprecision(2) << time << ".dat";

    std::ofstream file(filename.str());
    for (int i = 0; i <= mx; i++) {
        file << i * dx << " " << u_new(i) << "\n";
    }
    file.close();
}
