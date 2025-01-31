use image::{Rgb, RgbImage};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::fmt;
use std::mem;

// -------------------------
// Ray and HitRecord types
// -------------------------
#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

#[derive(Clone, Debug)]
struct HitRecord {
    t: f32,
    point: Vector3<f32>,
    normal: Vector3<f32>,
    color: Vector3<f32>, // diffuse reflectance (albedo), unitless (0-1)
}

// -------------------------
// AABB for BVH
// -------------------------
#[derive(Clone, Debug)]
struct AABB {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl AABB {
    fn hit(&self, ray: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        for a in 0..3 {
            let inv_d = 1.0 / ray.direction[a];
            let mut t0 = (self.min[a] - ray.origin[a]) * inv_d;
            let mut t1 = (self.max[a] - ray.origin[a]) * inv_d;
            if inv_d < 0.0 {
                mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }
}

fn surrounding_box(box0: &AABB, box1: &AABB) -> AABB {
    let small = Vector3::new(
        box0.min.x.min(box1.min.x),
        box0.min.y.min(box1.min.y),
        box0.min.z.min(box1.min.z),
    );
    let big = Vector3::new(
        box0.max.x.max(box1.max.x),
        box0.max.y.max(box1.max.y),
        box0.max.z.max(box1.max.z),
    );
    AABB {
        min: small,
        max: big,
    }
}

// -------------------------
// Hittable trait and cloning
// -------------------------
trait HittableClone {
    fn clone_box(&self) -> Box<dyn Hittable>;
}

impl<T> HittableClone for T
where
    T: 'static + Hittable + Clone,
{
    fn clone_box(&self) -> Box<dyn Hittable> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Hittable> {
    fn clone(&self) -> Box<dyn Hittable> {
        self.clone_box()
    }
}

trait Hittable: HittableClone + Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    fn bounding_box(&self) -> Option<AABB>;
}

// -------------------------
// Sphere
// -------------------------
#[derive(Clone)]
struct Sphere {
    center: Vector3<f32>,
    radius: f32,
    color: Vector3<f32>,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let half_b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();
        let mut root = (-half_b - sqrt_d) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrt_d) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }
        let point = ray.origin + ray.direction * root;
        let normal = (point - self.center) / self.radius;
        Some(HitRecord {
            t: root,
            point,
            normal,
            color: self.color,
        })
    }

    fn bounding_box(&self) -> Option<AABB> {
        let r_vec = Vector3::new(self.radius, self.radius, self.radius);
        Some(AABB {
            min: self.center - r_vec,
            max: self.center + r_vec,
        })
    }
}

// -------------------------
// Triangle using Möller–Trumbore
// -------------------------
#[derive(Clone)]
struct Triangle {
    v0: Vector3<f32>,
    v1: Vector3<f32>,
    v2: Vector3<f32>,
    color: Vector3<f32>,
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let epsilon = 1e-7;
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let h = ray.direction.cross(&edge2);
        let a = edge1.dot(&h);
        if a.abs() < epsilon {
            return None; // Ray is parallel to triangle.
        }
        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(&h);
        if u < 0.0 || u > 1.0 {
            return None;
        }
        let q = s.cross(&edge1);
        let v = f * ray.direction.dot(&q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * edge2.dot(&q);
        if t < t_min || t > t_max {
            return None;
        }
        let point = ray.origin + ray.direction * t;
        let normal = edge1.cross(&edge2).normalize();
        Some(HitRecord {
            t,
            point,
            normal,
            color: self.color,
        })
    }

    fn bounding_box(&self) -> Option<AABB> {
        let min_x = self.v0.x.min(self.v1.x).min(self.v2.x);
        let min_y = self.v0.y.min(self.v1.y).min(self.v2.y);
        let min_z = self.v0.z.min(self.v1.z).min(self.v2.z);
        let max_x = self.v0.x.max(self.v1.x).max(self.v2.x);
        let max_y = self.v0.y.max(self.v1.y).max(self.v2.y);
        let max_z = self.v0.z.max(self.v1.z).max(self.v2.z);
        let padding = 1e-4;
        Some(AABB {
            min: Vector3::new(min_x - padding, min_y - padding, min_z - padding),
            max: Vector3::new(max_x + padding, max_y + padding, max_z + padding),
        })
    }
}

// -------------------------
// BVH Node
// -------------------------
struct BVHNode {
    left: Box<dyn Hittable>,
    right: Box<dyn Hittable>,
    bbox: AABB,
}

// Custom Debug impl that prints only the bounding box.
impl fmt::Debug for BVHNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BVHNode").field("bbox", &self.bbox).finish()
    }
}

impl BVHNode {
    fn new(mut objects: Vec<Box<dyn Hittable>>) -> Self {
        // Choose a random axis (0 = x, 1 = y, 2 = z)
        let axis = rand::random::<usize>() % 3;
        objects.sort_by(|a, b| {
            let box_a = a.bounding_box().expect("No bounding box");
            let box_b = b.bounding_box().expect("No bounding box");
            box_a.min[axis]
                .partial_cmp(&box_b.min[axis])
                .unwrap_or(Ordering::Equal)
        });

        let n = objects.len();
        if n == 1 {
            let object = objects.remove(0);
            let bbox = object.bounding_box().expect("No bounding box");
            // Duplicate the object for both children.
            BVHNode {
                left: object.clone_box(),
                right: object,
                bbox,
            }
        } else if n == 2 {
            let right = objects.pop().unwrap();
            let left = objects.pop().unwrap();
            let box_left = left.bounding_box().expect("No bounding box");
            let box_right = right.bounding_box().expect("No bounding box");
            let bbox = surrounding_box(&box_left, &box_right);
            BVHNode { left, right, bbox }
        } else {
            let mid = n / 2;
            let right_vec = objects.split_off(mid);
            let left_node = BVHNode::new(objects);
            let right_node = BVHNode::new(right_vec);
            let box_left = left_node.bounding_box().expect("No bbox");
            let box_right = right_node.bounding_box().expect("No bbox");
            let bbox = surrounding_box(&box_left, &box_right);
            BVHNode {
                left: Box::new(left_node),
                right: Box::new(right_node),
                bbox,
            }
        }
    }
}

impl Clone for BVHNode {
    fn clone(&self) -> Self {
        BVHNode {
            left: self.left.clone_box(),
            right: self.right.clone_box(),
            bbox: self.bbox.clone(),
        }
    }
}

impl Hittable for BVHNode {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if !self.bbox.hit(ray, t_min, t_max) {
            return None;
        }
        let left_hit = self.left.hit(ray, t_min, t_max);
        let new_t_max = if let Some(ref rec) = left_hit {
            rec.t
        } else {
            t_max
        };
        let right_hit = self.right.hit(ray, t_min, new_t_max);
        right_hit.or(left_hit)
    }

    fn bounding_box(&self) -> Option<AABB> {
        Some(self.bbox.clone())
    }
}

// -------------------------
// Light, Reservoir, and Shading
// -------------------------
struct Light {
    position: Vector3<f32>,
    intensity: Vector3<f32>, // Radiant intensity in W/sr
}

#[derive(Clone, Debug)]
struct LightSample {
    direction: Vector3<f32>,
    distance: f32,
    intensity: Vector3<f32>,
    pdf: f32,
}

fn sample_light(light: &Light, hit_point: &Vector3<f32>) -> LightSample {
    let to_light = light.position - hit_point;
    let distance = to_light.magnitude();
    let direction = to_light / distance;
    LightSample {
        direction,
        distance,
        intensity: light.intensity,
        pdf: 1.0,
    }
}

struct Reservoir<T> {
    sample: Option<T>,
    w_sum: f32,
    M: u32,
}

impl<T> Reservoir<T> {
    fn new() -> Self {
        Reservoir {
            sample: None,
            w_sum: 0.0,
            M: 0,
        }
    }
    fn update(&mut self, sample: T, weight: f32, rng: &mut impl Rng) {
        self.M += 1;
        self.w_sum += weight;
        if rng.gen::<f32>() < weight / self.w_sum {
            self.sample = Some(sample);
        }
    }
}

fn in_shadow(ray: &Ray, t_max: f32, world: &dyn Hittable) -> bool {
    world.hit(ray, 0.001, t_max).is_some()
}

fn shade(hit: &HitRecord, light: &Light, world: &dyn Hittable, rng: &mut impl Rng) -> Vector3<f32> {
    let num_candidates = 16;
    let mut reservoir = Reservoir::<LightSample>::new();
    for _ in 0..num_candidates {
        let ls = sample_light(light, &hit.point);
        let epsilon = 0.001;
        let shadow_origin = hit.point + hit.normal * epsilon;
        let shadow_ray = Ray {
            origin: shadow_origin,
            direction: ls.direction,
        };
        let occluded = in_shadow(&shadow_ray, ls.distance, world);
        let cos_theta = hit.normal.dot(&ls.direction).max(0.0);
        let weight = if occluded {
            0.0
        } else {
            cos_theta / (ls.distance * ls.distance)
        };
        reservoir.update(ls, weight, rng);
    }
    if let Some(sample) = reservoir.sample {
        let epsilon = 0.001;
        let shadow_origin = hit.point + hit.normal * epsilon;
        let shadow_ray = Ray {
            origin: shadow_origin,
            direction: sample.direction,
        };
        if in_shadow(&shadow_ray, sample.distance, world) {
            return Vector3::zeros();
        }
        let cos_theta = hit.normal.dot(&sample.direction).max(0.0);
        let contribution = sample.intensity * cos_theta / (sample.distance * sample.distance);
        // Lambertian BRDF: reflectance/π
        return hit.color.component_mul(&contribution) / std::f32::consts::PI;
    }
    Vector3::zeros()
}

// -------------------------
// Reference Scene (Physically Realistic)
// -------------------------
// All dimensions are in meters and light intensity in W/sr.
// This scene consists of:
//  • A floor (two triangles)
//  • A back wall (two triangles)
//  • A diffuse sphere (object)
//  • A point light located above the sphere.
fn reference_scene() -> Vec<Box<dyn Hittable>> {
    let mut objects: Vec<Box<dyn Hittable>> = Vec::new();

    // Floor: a rectangle from (-2,0,-5) to (2,0,0)
    let floor_color = Vector3::new(0.9, 0.9, 0.9);
    objects.push(Box::new(Triangle {
        v0: Vector3::new(-2.0, 0.0, -5.0),
        v1: Vector3::new(2.0, 0.0, -5.0),
        v2: Vector3::new(2.0, 0.0, 0.0),
        color: floor_color,
    }));
    objects.push(Box::new(Triangle {
        v0: Vector3::new(-2.0, 0.0, -5.0),
        v1: Vector3::new(2.0, 0.0, 0.0),
        v2: Vector3::new(-2.0, 0.0, 0.0),
        color: floor_color,
    }));

    // Back wall: vertical plane at z = -5, from (-2,0,-5) to (2,2,-5)
    let wall_color = Vector3::new(0.9, 0.9, 0.9);
    objects.push(Box::new(Triangle {
        v0: Vector3::new(-2.0, 0.0, -5.0),
        v1: Vector3::new(2.0, 0.0, -5.0),
        v2: Vector3::new(2.0, 2.0, -5.0),
        color: wall_color,
    }));
    objects.push(Box::new(Triangle {
        v0: Vector3::new(-2.0, 0.0, -5.0),
        v1: Vector3::new(2.0, 2.0, -5.0),
        v2: Vector3::new(-2.0, 2.0, -5.0),
        color: wall_color,
    }));

    // Diffuse sphere: center (0,0.5,-3) with radius 0.5
    let sphere_color = Vector3::new(0.8, 0.2, 0.2); // red-ish
    objects.push(Box::new(Sphere {
        center: Vector3::new(0.0, 0.5, -3.0),
        radius: 0.5,
        color: sphere_color,
    }));

    objects
}

// -------------------------
// Main: Build scene, BVH, and render
// -------------------------
fn main() {
    let width = 800;
    let height = 600;
    let mut img = RgbImage::new(width, height);

    // Use the reference scene with physically realistic geometry.
    let objects = reference_scene();
    let world = BVHNode::new(objects);

    // Define a point light.
    // In physical units: position in meters, intensity in W/sr.
    let light = Light {
        position: Vector3::new(0.0, 1.8, -3.0),
        intensity: Vector3::new(500.0, 500.0, 500.0), // e.g., a 500 W/sr source
    };

    // Camera parameters.
    // Place camera so that it sees the reference scene.
    let camera_pos = Vector3::new(0.0, 1.0, 1.0);
    let fov_deg: f32 = 45.0;
    let scale = (fov_deg.to_radians() * 0.5).tan();
    let aspect_ratio = width as f32 / height as f32;

    let mut rng = StdRng::from_entropy();
    let samples_per_pixel = 16;

    // Render loop.
    for j in 0..height {
        for i in 0..width {
            let mut pixel_color = Vector3::zeros();
            for _ in 0..samples_per_pixel {
                // Jitter for anti-aliasing.
                let u = (i as f32 + rng.gen::<f32>()) / width as f32;
                let v = (j as f32 + rng.gen::<f32>()) / height as f32;
                let x = (2.0 * u - 1.0) * aspect_ratio * scale;
                let y = (1.0 - 2.0 * v) * scale;
                // Compute ray direction in camera space.
                // Here we assume a simple pinhole camera looking toward -z.
                let ray_dir = Vector3::new(x, y, -1.0).normalize();
                let ray = Ray {
                    origin: camera_pos,
                    direction: ray_dir,
                };
                if let Some(hit) = world.hit(&ray, 0.001, std::f32::MAX) {
                    let sample_color = shade(&hit, &light, &world, &mut rng);
                    pixel_color += sample_color;
                } else {
                    // Background: use a low-level ambient radiance.
                    pixel_color += Vector3::new(0.02, 0.02, 0.02);
                }
            }
            pixel_color /= samples_per_pixel as f32;
            // Gamma correction.
            let gamma = 2.2;
            let r = pixel_color.x.max(0.0).min(1.0).powf(1.0 / gamma);
            let g = pixel_color.y.max(0.0).min(1.0).powf(1.0 / gamma);
            let b = pixel_color.z.max(0.0).min(1.0).powf(1.0 / gamma);
            img.put_pixel(
                i,
                j,
                Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
            );
        }
    }

    img.save("output.png").unwrap();
}
