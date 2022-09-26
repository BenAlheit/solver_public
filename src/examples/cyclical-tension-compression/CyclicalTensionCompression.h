
#ifndef SOLVER_CYCLICALTENSIONCOMPRESSION_H
#define SOLVER_CYCLICALTENSIONCOMPRESSION_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../../materials/OutputFlags.h"
#include "../../problem/FEMesh.h"
#include "../../Solver.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
class CyclicalTensionCompression {
public:
    CyclicalTensionCompression(const string &name,
                               const map<unsigned int, Material<dim> *> &materials,
                               const double &extension,
                               const double &compression,
                               const unsigned int &n_steps,
                               const unsigned int &n_out,
                               const vector<double> &dts,
                               const map<nScalarOutput, vector<unsigned int>, nOutputHash> & n_scalar_outputs = {},
                               const map<nVectorOutput, vector<unsigned int>, nOutputHash> & n_vector_outputs = {},
                               const map<nTensorOutput, vector<unsigned int>, nOutputHash> & n_tensor_outputs = {},
                               const unsigned int &order = 1,
                               const unsigned int &n_refinements = 2);

    Triangulation<dim> triangulation;
    vector<Stage<dim>> stages;
    Time time;

private:
    const string name;
    const double extension, compression;
    const unsigned int n_steps, n_out, n_refinements, order;
    const vector<double> &dts;
    const map<unsigned int, Material<dim> *> materials;
    array<unsigned int, dim> range;

    Point<dim> p1 = Point<dim>({0, 0, 0});
    Point<dim> p2 = Point<dim>({1, 1, 1});

    void make_mesh();
};

template<unsigned int dim>
CyclicalTensionCompression<dim>::CyclicalTensionCompression(const string &name,
                                                            const map<unsigned int, Material<dim> *> &materials,
                                                            const double &extension,
                                                            const double &compression,
                                                            const unsigned int &n_steps,
                                                            const unsigned int &n_out,
                                                            const vector<double> &dts,
                                                            const map<nScalarOutput, vector<unsigned int>, nOutputHash> & n_scalar_outputs,
                                                            const map<nVectorOutput, vector<unsigned int>, nOutputHash> & n_vector_outputs,
                                                            const map<nTensorOutput, vector<unsigned int>, nOutputHash> & n_tensor_outputs,
                                                            const unsigned int &order,
                                                            const unsigned int &n_refinements)
        : name(name), n_steps(n_steps), n_out(n_out), extension(extension), compression(compression), dts(dts),
          materials(materials), order(order), n_refinements(n_refinements) {
    iota(range.begin(), range.end(), 0);
    make_mesh();

    time = Time(/*end_time*/ dts.at(0),
                             n_steps,
                             n_out);

    vector<DirichletBoundaryCondition<dim> *> init_extension_dbcs, compression_dbcs, extension_dbcs;
    for (const auto &i: range) {
        init_extension_dbcs.push_back(new Slider<dim>(1 + i, i));
        compression_dbcs.push_back(new Slider<dim>(1 + i, i));
        extension_dbcs.push_back(new Slider<dim>(1 + i, i));
    }

    init_extension_dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 4,
            /*components*/ vector<unsigned int>({0}),
            /*magnitudes*/ vector<double>({extension})));

    compression_dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 4,
            /*components*/ vector<unsigned int>({0}),
            /*end values*/ vector<double>({-compression}),
            /*start values*/ vector<double>({extension})));

    extension_dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 4,
            /*components*/ vector<unsigned int>({0}),
            /*end values*/ vector<double>({extension}),
            /*start values*/ vector<double>({-compression})));

    vector<unsigned int> material_ids({0});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
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

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> averaged_scalar_outputs;
    averaged_scalar_outputs[OutputFlags::ScalarOutputFlag::ELASTIC_STRAIN_ENERGY] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> averaged_tensor_outputs;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    averaged_tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;


    stages.push_back(Stage<dim>(&time,
                                init_extension_dbcs,
                                vector<NeumannBoundaryCondition<dim> *>({}),
                                scalar_outputs,
                                vector_outputs,
                                tensor_outputs,
                                mesh_outputs,
                                averaged_scalar_outputs,
                                {},
                                averaged_tensor_outputs,
                                dts.at(0),
                                n_steps,
                                n_out,
                                n_scalar_outputs,
                                n_vector_outputs,
                                n_tensor_outputs));

    double end_time = dts.at(0);
    for (unsigned int i = 1; i < dts.size(); ++i) {
        end_time += dts.at(i);
        if (i % 2 == 0) { // tension
            stages.push_back(Stage<dim>(&time,
                                        extension_dbcs,
                                        vector<NeumannBoundaryCondition<dim> *>({}),
                                        scalar_outputs,
                                        vector_outputs,
                                        tensor_outputs,
                                        mesh_outputs,
                                        averaged_scalar_outputs,
                                        {},
                                        averaged_tensor_outputs,
                                        end_time,
                                        n_steps,
                                        n_out,
                                        n_scalar_outputs,
                                        n_vector_outputs,
                                        n_tensor_outputs));
        } else { //compression
            stages.push_back(Stage<dim>(&time,
                                        compression_dbcs,
                                        vector<NeumannBoundaryCondition<dim> *>({}),
                                        scalar_outputs,
                                        vector_outputs,
                                        tensor_outputs,
                                        mesh_outputs,
                                        averaged_scalar_outputs,
                                        {},
                                        averaged_tensor_outputs,
                                        end_time,
                                        n_steps,
                                        n_out,
                                        n_scalar_outputs,
                                        n_vector_outputs,
                                        n_tensor_outputs));
        }
    }

    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();
}

template<unsigned int dim>
void CyclicalTensionCompression<dim>::make_mesh() {
    GridGenerator::hyper_rectangle(triangulation, p1, p2);

    for (const auto &cell: triangulation.cell_iterators())
        cell->set_material_id(0);

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

    if (n_refinements > 0) triangulation.refine_global(n_refinements);

}


#endif //SOLVER_CYCLICALTENSIONCOMPRESSION_H
