#ifndef SOLVER_STAGE_H
#define SOLVER_STAGE_H

#include "BoundaryCondition.h"
#include "Time.h"
#include "../materials/OutputFlags.h"

using namespace OutputFlags;

template<unsigned int dim>
class Stage {
public:
//    explicit Stage(const Time *time) : time(time) {};
    explicit Stage(Time *time) : time(time) {};

//    Stage(const Time *time,
    Stage(Time *time,
          const vector<DirichletBoundaryCondition<dim> *> & in_dbcs,
          const vector<NeumannBoundaryCondition<dim> *> & in_nbcs,
          const map<ScalarOutputFlag, vector<unsigned int>> & scalar_outputs = {},
          const map<VectorOutputFlag, vector<unsigned int>> & vector_outputs = {},
          const map<TensorOutputFlag, vector<unsigned int>> & tensor_outputs = {},
          const vector<MeshOutputFlag> & mesh_outputs = {},
          const map<ScalarOutputFlag, vector<unsigned int>> & scalar_volume_average = {},
          const map<VectorOutputFlag, vector<unsigned int>> & vector_volume_average = {},
          const map<TensorOutputFlag, vector<unsigned int>> & tensor_volume_average = {},
          const double & end_time = -1,
          const unsigned int & n_steps = 0,
          const unsigned int & n_out = 0)
            : dbcs(in_dbcs)
            , nbcs(in_nbcs)
            , time(time)
            , scalar_outputs(scalar_outputs)
            , vector_outputs(vector_outputs)
            , tensor_outputs(tensor_outputs)
            , mesh_outputs(mesh_outputs)
            , scalar_volume_average(scalar_volume_average)
            , vector_volume_average(vector_volume_average)
            , tensor_volume_average(tensor_volume_average)
            , end_time(end_time<=0 ? time->end() : end_time)
            , n_steps(n_steps==0 ? time->get_n_steps() : n_steps)
            , n_out(n_out==0 ? time->get_n_steps_out() : n_out)
            {};

    vector<DirichletBoundaryCondition<dim> *> get_dbcs() const { return dbcs; }

    vector<NeumannBoundaryCondition<dim> *> get_nbcs() const { return nbcs; }

    void add_dbc(const DirichletBoundaryCondition<dim> *dbc) { dbcs.push_back(dbc); };

    void add_nbc(const NeumannBoundaryCondition<dim> *nbc) { nbcs.push_back(nbc); };

    const map<ScalarOutputFlag, vector<unsigned int>> & get_scalar_outputs() const {return scalar_outputs;};
    const map<VectorOutputFlag, vector<unsigned int>> & get_vector_outputs() const {return vector_outputs;};
    const map<TensorOutputFlag, vector<unsigned int>> & get_tensor_outputs() const {return tensor_outputs;};

    const map<nScalarOutput, vector<unsigned int>, nScalarOutputHash> & get_n_scalar_outputs() const
    {return n_scalar_outputs;};
    const map<nVectorOutput, vector<unsigned int>, nVectorOutputHash> & get_n_vector_outputs() const
    {return n_vector_outputs;};
    const map<nTensorOutput, vector<unsigned int>, nTensorOutputHash> & get_n_tensor_outputs() const
    {return n_tensor_outputs;};

    const vector<MeshOutputFlag> & get_mesh_outputs() const {return mesh_outputs;};

    const map<ScalarOutputFlag, vector<unsigned int>> & get_scalar_volume_average_outputs() const
    {return scalar_volume_average;};
    const map<VectorOutputFlag, vector<unsigned int>> & get_vector_volume_average_outputs() const
    {return vector_volume_average;};
    const map<TensorOutputFlag, vector<unsigned int>> & get_tensor_volume_average_outputs() const
    {return tensor_volume_average;};

    void update_time_values(){ time->next_stage(end_time, n_steps, n_out); };

protected:

private:
    Time *time;
//    const Time *time;
    const double end_time;
    const unsigned int n_steps, n_out;

    vector<DirichletBoundaryCondition<dim> *> dbcs;
    vector<NeumannBoundaryCondition<dim> *> nbcs;
    const map<ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    const map<VectorOutputFlag, vector<unsigned int>> vector_outputs;
    const map<TensorOutputFlag, vector<unsigned int>> tensor_outputs;

    const map<nScalarOutput, vector<unsigned int>, nScalarOutputHash> n_scalar_outputs;
    const map<nVectorOutput, vector<unsigned int>, nVectorOutputFlag> n_vector_outputs;
    const map<nTensorOutput, vector<unsigned int>, nTensorOutputHash> n_tensor_outputs;

    const map<ScalarOutputFlag, vector<unsigned int>> scalar_volume_average;
    const map<VectorOutputFlag, vector<unsigned int>> vector_volume_average;
    const map<TensorOutputFlag, vector<unsigned int>> tensor_volume_average;

    const vector<MeshOutputFlag> mesh_outputs;
};

#endif //SOLVER_STAGE_H
