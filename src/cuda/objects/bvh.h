#ifndef BVHH
#define BVHH

#include <curand_kernel.h>

#include "objects/hittable.h"
#include "objects/hittable_list.h"
#include "primitives/aabb.h"

#include <algorithm>

class bvh_node: public hittable {
    public:
        hittable *left;
        hittable *right;
        aabb bbox;

        __device__ bvh_node(hittable_list& list, curandState* local_rand_state) : bvh_node(list.objects, 0, list.objects_count, local_rand_state) {}

        __device__ bvh_node(hittable **objects, int start, int end, curandState* local_rand_state) {
            int axis = int(3.0f * curand_uniform(local_rand_state));
            
            int object_span = end - start;

            if (object_span == 1) {
                left = right = objects[start];
            } else if (object_span == 2) {
                if (box_compare(objects[start], objects[start+1], axis)) {
                    left = objects[start];
                    right = objects[start+1];
                } else {
                    left = objects[start+1];
                    right = objects[start];
                }
            } else {
                // Simple bubble sort for small arrays (more efficient than complex sorting in CUDA)
                for (int i = start; i < end - 1; i++) {
                    for (int j = start; j < end - 1 - (i - start); j++) {
                        if (!box_compare(objects[j], objects[j+1], axis)) {
                            hittable* temp = objects[j];
                            objects[j] = objects[j+1];
                            objects[j + 1] = temp;
                        }
                    }
                }

                int mid = start + object_span / 2.0f;
                left = new bvh_node(objects, start, mid, local_rand_state);
                right = new bvh_node(objects, mid, end, local_rand_state);
            }

            bbox = aabb(left->bounding_box(), right->bounding_box());
        }

        __device__ static bool box_compare(hittable* a, hittable* b, int axis_index) {
            auto a_axis_min = a->bounding_box().axis_min(axis_index);
            auto b_axis_min = b->bounding_box().axis_min(axis_index);
            return a_axis_min < b_axis_min;
        }

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual aabb bounding_box() const;
};


__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!bbox.hit(r, t_min, t_max)) return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ aabb bvh_node::bounding_box() const {
    return bbox;
}

#endif