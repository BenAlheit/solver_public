#ifndef SOLVER_VISCOCRYSTALPLASTICCUBEWITHSPHEREPBC_H
#define SOLVER_VISCOCRYSTALPLASTICCUBEWITHSPHEREPBC_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../../materials/OutputFlags.h"
#include "../../problem/FEMesh.h"
#include "../../utils/utils.h"
#include "../../Solver.h"


using namespace std;
using namespace dealii;


template<unsigned int dim>
class ViscoCrystalplasticCubeWithSpherePBC {
public:
    explicit ViscoCrystalplasticCubeWithSpherePBC(unsigned int n_refinements = 2);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;
//    FEMesh<dim> mesh;
protected:

private:

    void read_and_refine_mesh();

    const unsigned int n_refinements;

    Point<dim> p1 = Point<dim>({-0.5, -0.5, 0});
    Point<dim> p2 = Point<dim>({0.5, 0.5, 1});
    array<unsigned int, dim> range;
    const unsigned int order = 1;
};

template<unsigned int dim>
ViscoCrystalplasticCubeWithSpherePBC<dim>::ViscoCrystalplasticCubeWithSpherePBC(unsigned int n_refinements):
        n_refinements(n_refinements) {

    iota(range.begin(), range.end(), 0);
    read_and_refine_mesh();

    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    auto elastic_base = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    const unsigned int n_sys = 1;
    const double visc = 500;
    const double m = 0.5;
    const double sig_y = 60;
    const double q = 0.5;
    const double rot_deg = 45;

    const string name = "vcp-1-cube-particle-psc-"+ to_string((int)rot_deg)+"-deg";

    vector<double> ref_euler_angles(dim, 0);
    ref_euler_angles.at(2) = rot_deg;

    auto example_state = new SingleSlipCrystalPlasticityState<dim>(ref_euler_angles);

    materials[/*material id*/ 0] = new ExplicitRateCrystalDependentPlasticity<dim>(elastic_base,
                                                                                   example_state,
                                                                                   sig_y,
                                                                                   visc,
                                                                                   m,
                                                                                   q);


    materials[/*material id*/ 1] = new NeoHookean<dim>(/*material id*/ 1, kappa, mu);
    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    unsigned int n_steps = 200;
    unsigned int n_out = 25;
    double mag = -0.25;

    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    typedef pair<unsigned int, unsigned int> b_pair;
    vector<b_pair> b_pairs({b_pair(1, 4),
                            b_pair(2, 5),
                            b_pair(3, 6)});

    Tensor<2, dim> Grad_u_end;
//    Grad_u_end[0][1] = mag;

    Grad_u_end[1][1] = mag;
    Grad_u_end[0][0] = 1/(1+mag) - 1;

    PeriodicBoundaryCondition<dim> pbc(mesh.get_dof_handler(),
                                       mesh.get_triangulation(),
                                       b_pairs,
                                       Grad_u_end);

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    vector<unsigned int> material_ids({0, 1});
    vector<unsigned int> viscoplastic_material_ids({0});

    scalar_outputs[OutputFlags::ScalarOutputFlag::J] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::P] = material_ids;

    map<OutputFlags::VectorOutputFlag, vector<unsigned int>> vector_outputs;
    vector_outputs[OutputFlags::VectorOutputFlag::NB1] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB2] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB3] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> tensor_outputs;
    tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRAIN] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FP] = viscoplastic_material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FE] = viscoplastic_material_ids;

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> averaged_scalar_outputs;
    averaged_scalar_outputs[OutputFlags::ScalarOutputFlag::ELASTIC_STRAIN_ENERGY] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> averaged_tensor_outputs;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::FirstPiolaStress] = material_ids;


    map<nScalarOutput, vector<unsigned int>, nOutputHash> n_scalar_outputs;
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::H, n_sys), viscoplastic_material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::ta, n_sys), viscoplastic_material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::nu, n_sys), viscoplastic_material_ids});


    map<nVectorOutput, vector<unsigned int>, nOutputHash> n_vector_outputs;
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::M, n_sys), viscoplastic_material_ids});
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::S, n_sys), viscoplastic_material_ids});

    map<nTensorOutput, vector<unsigned int>, nOutputHash> n_tensor_outputs;
    n_tensor_outputs.insert({OutputFlags::nTensorOutput(OutputFlags::nTensorOutputFlag::SYS, n_sys), viscoplastic_material_ids});

    Stage<dim> first_stage(&time,
                           vector<DirichletBoundaryCondition<dim> *>({&pbc}),
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs,
                           averaged_scalar_outputs,
                           {},
                           averaged_tensor_outputs,
                           time.end(),
                           time.get_n_steps(),
                           time.get_n_steps_out(),
                           n_scalar_outputs,
                           n_vector_outputs,
                           n_tensor_outputs);

    stages.push_back(first_stage);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();
}

template<unsigned int dim>
void ViscoCrystalplasticCubeWithSpherePBC<dim>::read_and_refine_mesh() {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    ifstream input_file("../src/examples/cube-with-sphere/cube-with-particle.ucd");
    grid_in.read_ucd(input_file);
    cout << "Read mesh" << endl;
    triangulation.reset_all_manifolds();
    triangulation.set_all_manifold_ids(0);
    Point<dim> centre(0.0, 0.0, 0.5);
    for (const auto &cell: triangulation.cell_iterators()) {
        if (cell->material_id() == 1) {
            for (const auto &face: cell->face_iterators()) {
                bool face_at_sphere_boundary = true;
                for (const auto v: face->vertex_indices()) {
                    if (fabs((face->vertex(v) - centre).norm() - 0.25) > 1e-3) {
                        face_at_sphere_boundary = false;
                        break;
                    }
                }
                if (face_at_sphere_boundary)
                    face->set_all_manifold_ids(1);
            }
        }
    }

    Point<dim> cell_centre;

    for (const auto &cell: triangulation.cell_iterators()) {
        cell_centre = cell->center();
        if ((cell_centre - centre).norm() > 0.25) {
            cell->set_material_id(0);
        } else {
            cell->set_material_id(1);
        }
    }

    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            for (const auto &i: range) {
                if (almost_equals(face->center()[i], p1[i])) {
                    face->set_boundary_id(1 + i);
                    break;
                } else if (almost_equals(face->center()[i], p2[i])) {
                    face->set_boundary_id(1 + dim + i);
                    break;
                }
            }
        }
    }

    triangulation.set_manifold(1, SphericalManifold<dim>(centre));
    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(triangulation);
    triangulation.set_manifold(0, transfinite_manifold);

    if (n_refinements > 0) triangulation.refine_global(n_refinements);

    cout << "Refined mesh" << endl;


    ofstream out("cube-with-sphere-" + to_string(n_refinements) + ".vtk");
    GridOut grid_out;
    grid_out.write_vtk(triangulation, out);

    cout << "Written mesh" << endl;

}

#endif //SOLVER_VISCOCRYSTALPLASTICCUBEWITHSPHEREPBC_H
