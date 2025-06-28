#ifndef HITTABLELISTH
#define HITTABLELISTH

#include "objects/hittable.h"
#include "primitives/aabb.h"

class hittable_list : public hittable {
    public:
        hittable **objects;
        int objects_count;
        aabb bbox;
        hittable *single_object; // For storing a single object (like BVH node)

        __device__ hittable_list() {}
        __device__ hittable_list(hittable **l, int n) {
            objects = l; objects_count = n;
            single_object = nullptr;
            bbox = objects[0]->bounding_box();
            for (int i = 1; i < objects_count; i++) {
                bbox = aabb(bbox, objects[i]->bounding_box());
            }
        }
        __device__ hittable_list(hittable *single_obj) {
            objects = nullptr; 
            objects_count = 1;
            single_object = single_obj;
            bbox = single_object->bounding_box();
        }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual aabb bounding_box() const;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // If we have a single object (like BVH node), just hit that
    if (single_object != nullptr) {
        return single_object->hit(r, t_min, t_max, rec);
    }
    
    // Otherwise, handle the array of objects as before
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < objects_count; i++) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ aabb hittable_list::bounding_box() const {
    return bbox;
}

#endif
