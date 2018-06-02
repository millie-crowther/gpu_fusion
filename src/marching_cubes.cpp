#include "marching_cubes.hpp"

#include <fstream>

void save_mesh(std::string model_name, int frame){
    std::cout << "Saving SDF to mesh..." << std::endl;

    std::string padding = "";
    if (frame < 1000) padding = "0" + padding;
    if (frame <  100) padding = "0" + padding;
    if (frame <   10) padding = "0" + padding;

    std::string filename = "frame" + padding + std::to_string(frame);

    // full filename
    std::string full_name = "../data/mesh/" + model_name + "/" + filename;

    // create triangles
    mesh_t mesh;
    create_mesh(0, &mesh);

    // create default material file
    std::ofstream mat_file;
    mat_file.open(full_name + ".mtl");
    mat_file <<
    "# Material file for " << filename  << std::endl <<
    "newmtl " << filename << "Material" << std::endl <<
    "Ns 96.078431"                      << std::endl <<
    "Ka 1.000000 1.000000 1.000000"     << std::endl <<
    "Kd 0.640000 0.640000 0.640000"     << std::endl <<
    "Ks 0.500000 0.500000 0.500000"     << std::endl <<
    "Ke 0.000000 0.000000 0.000000"     << std::endl <<
    "Ni 1.000000"                       << std::endl <<
    "d 1.000000"                        << std::endl <<
    "illum 2"                           << std::endl;
    mat_file.close();

    // save to wavefront .obj format
    std::ofstream mesh_file;
    mesh_file.open(full_name + ".obj");
    mesh_file <<
    "# Geometry file for " << filename << std::endl <<
    "mtllib " << filename << ".mtl"    << std::endl <<
    "o " << filename << "Object"       << std::endl;

    // vertices
    for (auto tri : mesh){
        for (int i = 0; i < 3; i++){
            float3 p = (tri.vertices[i] - size / 2.0f) / -100.0f;
            mesh_file << "v " << p.x << " " << p.y << " " << p.z << std::endl;
        }
    }

    // normals 
    for (auto tri : mesh){
        for (int i = 0; i < 3; i++){
            mesh_file << "vn ";

            for (int j = 0; j < 3; j++){
                mesh_file << normal(tri.vertices[i])[j] << " ";
            }
            mesh_file << std::endl;
        }
    }

    mesh_file <<
    "usemtl " << filename << "Material" << std::endl <<
    "s off"                             << std::endl;

    // faces
    for (int i = 0; i < mesh.size() * 3; i += 3){
        mesh_file << "f ";
        for (int j = 0; j < 3; j++){
            mesh_file << i+j+1 << "//" << i+j+1 << " ";
        }
        mesh_file << std::endl;
    }

    mesh_file.close();
}
