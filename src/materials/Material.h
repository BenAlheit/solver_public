#ifndef SOLVER_MATERIAL_H
#define SOLVER_MATERIAL_H

#include <deal.II/base/tensor.h>
#include "MaterialState.h"
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "../utils/utils.h"

using namespace dealii;
using namespace std;

template<unsigned int dim>
class Material {
public:
    Material(unsigned int id,
             const Tensor<2, dim> &orientation = Physics::Elasticity::StandardTensors<dim>::I)
            : id(id), orientation(orientation) {
        iota(dim_range.begin(), dim_range.end(), 0);
        for (const auto &i: dim_range)
            for (const auto &j: dim_range)
                for (const auto &k: dim_range)
                    for (const auto &l: dim_range)
                        IoI[i][j][k][l] = 0.5 * ((i == k ? 1 : 0) * (j == l ? 1 : 0) +
                                                 (i == l ? 1 : 0) * (j == k ? 1 : 0));
    };

    virtual void update_stress(const double &dt) = 0;

    virtual void update_tangent(const double &dt) = 0;

    virtual void update_stress_and_tangent(const double &dt) = 0;

    unsigned int get_id() const { return id; };

//    virtual StateBase<dim> *get_state() = 0;

    virtual void set_state(StateBase<dim> *new_state) = 0;

    virtual StateBase<dim> *create_state() const = 0;

    template<class state_type>
    void approximate_tangent(const double &dt,
                             state_type* current_state,
                             const double & eps = 1e-7);

protected:
    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;
    SymmetricTensor<4, dim> IoI;

    StateBase<dim> *state;
    array<unsigned int, dim> dim_range;

private:
    const unsigned int id;
};

template<unsigned int dim>
template<class state_type>
void Material<dim>::approximate_tangent(const double &dt,
                                        state_type* current_state,
                                        const double & eps) {
    set_state(current_state);
    update_stress(dt);

    Tensor<2, dim> tau, Delta_L, c_mn;

    tau = current_state->tau_n1;

//    double eps = 1e-7;

    typedef pair<unsigned int, unsigned int> p;
    vector<p> i_s({p(0, 0),
                   p(1, 1),
                   p(2, 2),
                   p(0, 1),
                   p(0, 2),
                   p(1, 2)});


//    state_type* working_state = current_state->copy();
    auto working_state = current_state->copy();
    working_state->copy(current_state);

    set_state(working_state);
    current_state->c_n1 = 0;
    for (const auto & v : i_s) {
        Delta_L = 0;
        Delta_L[v.first][v.second] += 1;
        Delta_L[v.second][v.first] += 1;
        Delta_L *= eps/2.;

        working_state->F_n1 = working_state->F_n1 + Delta_L * working_state->F_n1;
        update_stress(dt);
        c_mn = (working_state->tau_n1 - tau - Delta_L * tau - tau * transpose(Delta_L) ) / eps;

        for(const auto & i: dim_range)
            for(const auto & j: dim_range){
                current_state->c_n1[i][j][v.first][v.second] = c_mn[i][j];
                current_state->c_n1[j][i][v.first][v.second] = c_mn[i][j];
            }

        working_state->copy(current_state);
    }
    set_state(current_state);

    delete working_state;
}


template<unsigned int dim>
class ElasticMaterial : public Material<dim> {
public:
    explicit ElasticMaterial(unsigned int id,
                             const Tensor<2, dim> &orientation = Physics::Elasticity::StandardTensors<dim>::I)
            : Material<dim>(id, orientation) {};

    typedef ElasticState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type()); }

protected:
    state_type *state;

private:

};

template<unsigned int dim>
class IsoVolSplitElasticMaterial : public ElasticMaterial<dim> {
public:

    typedef IsoVolElasticState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type()); }

protected:
    state_type *state;

private:
};


template<unsigned int dim>
class HistoryDependentMaterial : public Material<dim> {

};


#endif //SOLVER_MATERIAL_H
