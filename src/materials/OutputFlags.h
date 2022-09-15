//
// Created by alhei on 2022/08/31.
//

#ifndef SOLVER_OUTPUTFLAGS_H
#define SOLVER_OUTPUTFLAGS_H

#include "../Exceptions.h"
#include <string>

using namespace std;


namespace OutputFlags{
    enum ScalarOutputFlag{
        P, J, EP, DLAM, FPLAS, ELASTIC_STRAIN_ENERGY
    };

    enum VectorOutputFlag{
        NS1, NS2, NS3, NB1, NB2, NB3
    };

    enum TensorOutputFlag{
        F, STRESS, STRAIN, T, FP, FE, FirstPiolaStress
    };

    enum MeshOutputFlag{
        MATERIAL_ID, BOUNDARY_ID, ELEMENT_QUALITY
    };

    string to_string(const MeshOutputFlag & flag){
        switch (flag) {
            case MATERIAL_ID:
                return "Material_ID";
            case BOUNDARY_ID:
                return "Boundary_ID";
            case ELEMENT_QUALITY:
                return "Element_quality";
            default:
                throw NotImplemented("to_string function not implemented for this MeshOutputFlag");
        }
    }

    string to_string(const ScalarOutputFlag & flag){
        switch (flag) {
            case J:
                return "Jacobian";
            case P:
                return "Pressure";
            case EP:
                return "Equivalent_plastic_strain";
            case DLAM:
                return "Plastic_strain_rate";
            case FPLAS:
                return "Plastic_yield_surface";
            case ELASTIC_STRAIN_ENERGY:
                return "Elastic_strain_energy";
            default:
                throw NotImplemented("to_string function not implemented for this ScalarOutputFlag");
        }
    }

    template<unsigned int dim>
    vector<string> create_string_vector_for_vector(const string & name){
        vector<string> out;
        array<unsigned int, dim> range;
        iota(range.begin(), range.end(), 0);
        for (const auto i: range)
            out.push_back(name + std::to_string(i + 1));
        return out;
    }

    template<unsigned int dim>
    vector<string> to_string(const VectorOutputFlag & flag){
        switch (flag) {
            case NS1:
                return create_string_vector_for_vector<dim>("PrincipleStress1");
            case NS2:
                return create_string_vector_for_vector<dim>("PrincipleStress2");
            case NS3:
                return create_string_vector_for_vector<dim>("PrincipleStress3");
            case NB1:
                return create_string_vector_for_vector<dim>("PrincipleStretch1");
            case NB2:
                return create_string_vector_for_vector<dim>("PrincipleStretch2");
            case NB3:
                return create_string_vector_for_vector<dim>("PrincipleStretch3");
            default:
                throw NotImplemented("to_string function not implemented for this VectorOutputFlag");
        }
    }

    template<unsigned int dim>
    vector<string> create_string_vector_for_tensor(const string & name){
        vector<string> out;
        array<unsigned int, dim> range;
        iota(range.begin(), range.end(), 0);
        for (const auto i: range)
            for (const auto j: range)
                out.push_back(name + std::to_string(i + 1) + std::to_string(j + 1));
        return out;
    }

    template<unsigned int dim>
    vector<string> to_string(const TensorOutputFlag & flag){
        switch (flag) {
            case F:
                return create_string_vector_for_tensor<dim>("DeformationGradient");
            case FE:
                return create_string_vector_for_tensor<dim>("ElasticDeformationGradient");
            case FP:
                return create_string_vector_for_tensor<dim>("PlasticDeformationGradient");
            case STRESS:
                return create_string_vector_for_tensor<dim>("Stress");
            case FirstPiolaStress:
                return create_string_vector_for_tensor<dim>("FirstPiolaStress");
            case STRAIN:
                return create_string_vector_for_tensor<dim>("Strain");
            default:
                throw NotImplemented("to_string function not implemented for this TensorOutputFlag");
        }
    }

}




#endif //SOLVER_OUTPUTFLAGS_H
