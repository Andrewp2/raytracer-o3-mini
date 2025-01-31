use image::{Rgb, RgbImage};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::fmt;
use std::mem;

const DEBUG_NORMALS: bool = false;

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
    color: Vector3<f32>, // diffuse albedo (0-1)
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
            return None; // Ray is parallel.
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
// Quad helper function
// -------------------------
// Given four vertices (in order), returns two triangles.
// The vertices should be ordered so that the computed normal points
// in the desired direction.
fn quad(
    v0: Vector3<f32>,
    v1: Vector3<f32>,
    v2: Vector3<f32>,
    v3: Vector3<f32>,
    color: Vector3<f32>,
) -> Vec<Box<dyn Hittable>> {
    // Two triangles: (v0, v1, v2) and (v0, v2, v3)
    vec![
        Box::new(Triangle { v0, v1, v2, color }),
        Box::new(Triangle {
            v0,
            v1: v2,
            v2: v3,
            color,
        }),
    ]
}

// -------------------------
// BVH Node
// -------------------------
struct BVHNode {
    left: Box<dyn Hittable>,
    right: Box<dyn Hittable>,
    bbox: AABB,
}

impl fmt::Debug for BVHNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BVHNode").field("bbox", &self.bbox).finish()
    }
}

impl BVHNode {
    fn new(mut objects: Vec<Box<dyn Hittable>>) -> Self {
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
// Light, Direct Illumination, and Sampling
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

fn direct_light(
    hit: &HitRecord,
    light: &Light,
    world: &dyn Hittable,
    rng: &mut impl Rng,
) -> Vector3<f32> {
    let num_samples = 16;
    let mut sum = Vector3::zeros();
    for _ in 0..num_samples {
        let ls = sample_light(light, &hit.point);
        let epsilon = 0.001;
        let shadow_origin = hit.point + hit.normal * epsilon;
        let shadow_ray = Ray {
            origin: shadow_origin,
            direction: ls.direction,
        };
        if !in_shadow(&shadow_ray, ls.distance, world) {
            let cos_theta = hit.normal.dot(&ls.direction).max(0.0);
            sum += ls.intensity * cos_theta / (ls.distance * ls.distance);
        }
    }
    sum /= num_samples as f32;
    hit.color.component_mul(&sum) / std::f32::consts::PI
}

fn in_shadow(ray: &Ray, t_max: f32, world: &dyn Hittable) -> bool {
    world.hit(ray, 0.001, t_max).is_some()
}

fn cosine_weighted_sample_hemisphere(normal: &Vector3<f32>, rng: &mut impl Rng) -> Vector3<f32> {
    let r1: f32 = rng.gen();
    let r2: f32 = rng.gen();
    let r = r1.sqrt();
    let phi = 2.0 * std::f32::consts::PI * r2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - r1).sqrt();
    let w = *normal;
    let a = if w.x.abs() > 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let v = w.cross(&a).normalize();
    let u = w.cross(&v);
    u * x + v * y + w * z
}

fn radiance(
    ray: &Ray,
    world: &dyn Hittable,
    light: &Light,
    rng: &mut impl Rng,
    depth: u32,
) -> Vector3<f32> {
    if depth == 0 {
        return Vector3::zeros();
    }
    if let Some(hit) = world.hit(ray, 0.001, std::f32::MAX) {
        if DEBUG_NORMALS {
            return (hit.normal + Vector3::new(1.0, 1.0, 1.0)) * 0.5;
        }
        let direct = direct_light(&hit, light, world, rng);
        let new_dir = cosine_weighted_sample_hemisphere(&hit.normal, rng);
        let new_origin = hit.point + hit.normal * 0.001;
        let indirect = radiance(
            &Ray {
                origin: new_origin,
                direction: new_dir,
            },
            world,
            light,
            rng,
            depth - 1,
        );
        return direct + hit.color.component_mul(&indirect);
    } else {
        return Vector3::new(0.02, 0.02, 0.02);
    }
}

// -------------------------
// Reference Scene (Physically Realistic)
// -------------------------
fn reference_scene() -> Vec<Box<dyn Hittable>> {
    let mut objects: Vec<Box<dyn Hittable>> = Vec::new();

    // Create floor quad.
    let floor_color = Vector3::new(0.9, 0.9, 0.9);
    let floor_v0 = Vector3::new(-2.0, 0.0, -5.0);
    let floor_v1 = Vector3::new(-2.0, 0.0, 0.0);
    let floor_v2 = Vector3::new(2.0, 0.0, 0.0);
    let floor_v3 = Vector3::new(2.0, 0.0, -5.0);
    objects.extend(quad(floor_v0, floor_v1, floor_v2, floor_v3, floor_color));

    // Create back wall quad.
    // Use the natural counterclockwise order so that the computed normal is (0,0,1).
    let wall_color = Vector3::new(0.9, 0.9, 0.9);
    let wall_v0 = Vector3::new(-2.0, 0.0, -5.0);
    let wall_v1 = Vector3::new(2.0, 0.0, -5.0);
    let wall_v2 = Vector3::new(2.0, 2.0, -5.0);
    let wall_v3 = Vector3::new(-2.0, 2.0, -5.0);
    objects.extend(quad(wall_v0, wall_v1, wall_v2, wall_v3, wall_color));

    // Diffuse sphere.
    let sphere_color = Vector3::new(0.8, 0.2, 0.2);
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

    let objects = reference_scene();
    let world = BVHNode::new(objects);

    let light = Light {
        position: Vector3::new(1.5, 1.5, -0.5),
        intensity: Vector3::new(50.0, 50.0, 50.0),
    };

    let camera_pos = Vector3::new(0.0, 1.0, 1.0);
    let fov_deg: f32 = 45.0;
    let scale = (fov_deg.to_radians() * 0.5).tan();
    let aspect_ratio = width as f32 / height as f32;

    let mut rng = StdRng::from_entropy();
    let samples_per_pixel = 16;
    let max_depth = 3;

    for j in 0..height {
        for i in 0..width {
            let mut pixel_color = Vector3::zeros();
            for _ in 0..samples_per_pixel {
                let u = (i as f32 + rng.gen::<f32>()) / width as f32;
                let v = (j as f32 + rng.gen::<f32>()) / height as f32;
                let x = (2.0 * u - 1.0) * aspect_ratio * scale;
                let y = (1.0 - 2.0 * v) * scale;
                let ray_dir = Vector3::new(x, y, -1.0).normalize();
                let ray = Ray {
                    origin: camera_pos,
                    direction: ray_dir,
                };
                pixel_color += radiance(&ray, &world, &light, &mut rng, max_depth);
            }
            pixel_color /= samples_per_pixel as f32;
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
