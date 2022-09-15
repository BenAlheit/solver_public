#ifndef SOLVER_NEOHOOKEAN_H
#define SOLVER_NEOHOOKEAN_H

#include "Material.h"
#include "MaterialState.h"

template<unsigned int dim>
class NeoHookean : public ElasticMaterial<dim> {
public:

    NeoHookean(unsigned int id, double kappa, double mu)
            : ElasticMaterial<dim>(id), kappa(kappa), mu(mu) {};

    void update_stress(const double &dt) override;

    void update_tangent(const double &dt) override;

    void update_stress_and_tangent(const double &dt) override;

protected:

private:
    double kappa, mu;
};

template<unsigned int dim>
void NeoHookean<dim>::update_stress(const double &dt) {
    double J = determinant(this->state->F_n1);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * this->state->F_n1;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);

    this->state->tau_n1 = mu * (Bbar - I1bar * this->I / 3.) + kappa * (pow(J, 2) - 1) * this->I / 2.;
    this->state->elastic_strain_energy_n1 = (mu * (I1bar - 3) + kappa * ((pow(J, 2) - 1)/2. - log(J)))/2.;
}

template<unsigned int dim>
void NeoHookean<dim>::update_tangent(const double &dt) {
    double J = determinant(this->state->F_n1);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * this->state->F_n1;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);

    this->state->c_n1 = 2 * mu * (I1bar * this->IoI / 3.
                                  - (outer_product(this->I, Bbar) + outer_product(Bbar, this->I)) / 3.
                                  + I1bar * this->IxI / 9.);
    this->state->c_n1 += kappa * (J * J * (this->IxI - this->IoI) + this->IoI);
    this->state->elastic_strain_energy_n1 = (mu * (I1bar - 3) + kappa * ((pow(J, 2) - 1)/2. - log(J)))/2.;

}

template<unsigned int dim>
void NeoHookean<dim>::update_stress_and_tangent(const double &dt) {
    double J = determinant(this->state->F_n1);
    Tensor<2, dim> Fbar = pow(J, -1. / 3.) * this->state->F_n1;
    SymmetricTensor<2, dim> Bbar = symmetrize(Fbar * transpose(Fbar));
    double I1bar = trace(Bbar);

    this->state->tau_n1 = mu * (Bbar - I1bar * this->I / 3.) + kappa * (pow(J, 2) - 1) * this->I / 2.;

    this->state->c_n1 = 2 * mu * (I1bar * this->IoI / 3.
                                  - (outer_product(this->I, Bbar) + outer_product(Bbar, this->I)) / 3.
                                  + I1bar * this->IxI / 9.);
    this->state->c_n1 += kappa * (J * J * (this->IxI - this->IoI) + this->IoI);
    this->state->elastic_strain_energy_n1 = (mu * (I1bar - 3) + kappa * ((pow(J, 2) - 1)/2. - log(J)))/2.;

}

#endif //SOLVER_NEOHOOKEAN_H
