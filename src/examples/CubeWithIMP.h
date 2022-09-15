#ifndef SOLVER_CUBEWITHIMP_H
#define SOLVER_CUBEWITHIMP_H

#include <boost/archive/text_iarchive.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../materials/OutputFlags.h"
#include "../problem/FEMesh.h"
#include "../Solver.h"

using namespace std;
using namespace dealii;

//bool almost_equals(double first, double second, double tol = 1e-7) {
//    return fabs(first - second) < tol;
//}

template<unsigned int dim>
class CubeWithIMP {
public:
    explicit CubeWithIMP (unsigned int n_refinements = 0);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;
//    FEMesh<dim> mesh;
protected:

private:

    void read_and_refine_mesh();

    const unsigned int n_refinements;

//    Point<dim> p1 = Point<dim>({-0.5, -0.5, 0});
//    Point<dim> p2 = Point<dim>({0.5, 0.5, 1});
    array<unsigned int, dim> range;
    const unsigned int order = 1;
};

template<unsigned int dim>
CubeWithIMP<dim>::CubeWithIMP(unsigned int n_refinements):
        n_refinements(n_refinements) {

    iota(range.begin(), range.end(), 0);
    read_and_refine_mesh();
//    string name = "elastic-cube-with-particle";
    string name = "elastic-cube-with-imp-shear";
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    const double mult = 1e2;
//    const double mult = 1;

    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);
    materials[/*material id*/ 1] = new NeoHookean<dim>(/*material id*/ 1, mult * kappa, mult * mu);

    unsigned int n_steps = 50;
    unsigned int n_out = 25;
    double pull = 0.9;
    double shear = 0.9;

    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    Fix<dim> left_fix(/*boundary_id*/2);


    DirichletBoundaryCondition<dim> right( /*boundary_id*/ 5,
            /*components*/ vector<unsigned int>({0, 1, 2}),
            /*magnitudes*/ vector<double>({0, shear, 0}));

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    vector<unsigned int> material_ids({0, 1});
//    vector<unsigned int> material_ids({0});
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

    Stage<dim> first_stage(&time,
                           vector<DirichletBoundaryCondition<dim> *>({&left_fix, &right}),
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs);

    stages.push_back(first_stage);
    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void CubeWithIMP<dim>::read_and_refine_mesh() {
    ifstream input_file("../src/examples/archn.dtri");
    {
        boost::archive::text_iarchive ia(input_file);
        triangulation.load(ia, 0);
    }

    if (n_refinements > 0) triangulation.refine_global(n_refinements);
}

#endif //SOLVER_CUBEWITHIMP_H
