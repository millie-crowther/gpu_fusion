#ifndef FUSION_H
#define FUSION_H

#include <string>
#include <vector>

class fusion_t {
public:
    // constructors and destructors
    fusion_t();
    ~fusion_t();
    
    // main public method
    void fusion();

private:
    void load_filenames(std::vector<std::string> * fns, int frames);
    sdf_t get_sdf(std::string filename);
    void initialise(sdf_t sdf);
    void update(bool is_rigid, sdf_t sdf);
};

#endif
