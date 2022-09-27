//
// Created by alhei on 2022/09/05.
//

#ifndef SOLVER_UTILS_H
#define SOLVER_UTILS_H

#include "cmath"
#include <deal.II/base/tensor.h>

using namespace std;
using namespace dealii;

bool almost_equals(const double & first, const double & second, const double & tol = 1e-10) {
    return fabs(first - second) < tol;
}

template<unsigned int dim>
void print_symmetric_4th_order(const SymmetricTensor<4, dim> & in){
    typedef pair<unsigned int, unsigned int> p;
    vector<p> i_s({p(0, 0),
                   p(1, 1),
                   p(2, 2),
                   p(0, 1),
                   p(0, 2),
                   p(1, 2)});

    for (const auto & vi : i_s){
        for (const auto & vj : i_s)
            cout << in[vi.first][vi.second][vj.first][vj.second] << "\t";
        cout << endl;
    }
}
template<unsigned int dim>
void print_2nd_order(const Tensor<2, dim> & in){
    typedef pair<unsigned int, unsigned int> p;
    array<unsigned int, dim> range;
    iota(range.begin(), range.end(), 0);

    for (const auto & vi : range){
        for (const auto & vj : range)
            cout << in[vi][vj] << "\t";
        cout << endl;
    }
}

template<unsigned int dim>
void polar_decomposition(const Tensor<2, dim> & F,
                         Tensor<2, dim> & V,
                         Tensor<2, dim> & R){
    SymmetricTensor<2, dim> B = symmetrize(F * transpose(F));
    const auto eigs = eigenvectors(B);
    V = 0;
    for (int i = 0; i < dim; ++i) {
        V += sqrt(eigs[i].first) * outer_product(eigs[i].second, eigs[i].second);
    }
    R = invert(V) * F;
}

#endif //SOLVER_UTILS_H
