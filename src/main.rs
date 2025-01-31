use image::{Rgb, RgbImage};
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A simple ray with an origin and direction.
#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

/// A sphere with a center, radius, and diffuse color.
#[derive(Clone, Debug)]
struct Sphere {
    center: Vector3<f32>,
    radius: f32,
    color: Vector3<f32>,
}

impl Sphere {
    /// Returns the distance along the ray if there is an intersection.
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            None
        } else {
            let sqrt_disc = discriminant.sqrt();
            let t1 = (-b - sqrt_disc) / (2.0 * a);
            let t2 = (-b + sqrt_disc) / (2.0 * a);
            if t1 > 0.001 {
                Some(t1)
            } else if t2 > 0.001 {
                Some(t2)
            } else {
                None
            }
        }
    }
}

/// A point light with a position and intensity.
#[derive(Clone, Debug)]
struct Light {
    position: Vector3<f32>,
    intensity: Vector3<f32>,
}

/// The scene contains a list of spheres and a single light.
struct Scene {
    spheres: Vec<Sphere>,
    light: Light,
}

/// A candidate sample from the light.
#[derive(Clone, Debug)]
struct LightSample {
    direction: Vector3<f32>,
    distance: f32,
    intensity: Vector3<f32>,
    pdf: f32,
}

/// For a point light the candidate is simply the direction from the hit point to the light.
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

/// A simple reservoir data structure.
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
    /// Updates the reservoir with a new candidate sample.
    /// This follows the reservoir sampling update: for each candidate with weight w,
    /// with probability w/(w_sum + w) the candidate replaces the current sample.
    fn update(&mut self, sample: T, weight: f32, rng: &mut impl Rng) {
        self.M += 1;
        self.w_sum += weight;
        if rng.gen::<f32>() < weight / self.w_sum {
            self.sample = Some(sample);
        }
    }
}

/// Checks if the light sample is occluded by any sphere.
/// The ray is cast from hit_point (offset by a small epsilon along the normal)
/// toward the light. If any intersection occurs before reaching the light, the sample is occluded.
fn in_shadow(
    hit_point: &Vector3<f32>,
    normal: &Vector3<f32>,
    ls: &LightSample,
    scene: &Scene,
) -> bool {
    let epsilon = 0.001;
    let shadow_origin = *hit_point + *normal * epsilon;
    let shadow_ray = Ray {
        origin: shadow_origin,
        direction: ls.direction,
    };
    for sphere in &scene.spheres {
        if let Some(t) = sphere.intersect(&shadow_ray) {
            if t < ls.distance {
                return true;
            }
        }
    }
    false
}

/// Traces a ray through the scene and, if a sphere is hit, returns:
/// (distance, hit point, surface normal, sphere color).
fn trace_ray(ray: &Ray, scene: &Scene) -> Option<(f32, Vector3<f32>, Vector3<f32>, Vector3<f32>)> {
    let mut closest = std::f32::MAX;
    let mut hit_color = Vector3::zeros();
    let mut hit_normal = Vector3::zeros();
    let mut hit_point = Vector3::zeros();
    let mut hit = false;
    for sphere in &scene.spheres {
        if let Some(t) = sphere.intersect(ray) {
            if t < closest {
                closest = t;
                hit_point = ray.origin + ray.direction * t;
                hit_normal = (hit_point - sphere.center).normalize();
                hit_color = sphere.color;
                hit = true;
            }
        }
    }
    if hit {
        Some((closest, hit_point, hit_normal, hit_color))
    } else {
        None
    }
}

/// Shades the hit point using reservoir-based importance sampling.
/// For a number of candidate light samples, we update a reservoir based on the sample weight,
/// and then use the chosen candidate to compute the final contribution.
/// Here the weight is computed (for diffuse surfaces) as cosθ/distance².
fn shade(
    hit_point: &Vector3<f32>,
    normal: &Vector3<f32>,
    color: &Vector3<f32>,
    scene: &Scene,
    rng: &mut impl Rng,
) -> Vector3<f32> {
    let num_candidates = 16;
    let mut reservoir = Reservoir::<LightSample>::new();

    for _ in 0..num_candidates {
        let ls = sample_light(&scene.light, hit_point);
        let occluded = in_shadow(hit_point, normal, &ls, scene);
        let cos_theta = normal.dot(&ls.direction).max(0.0);
        let weight = if occluded {
            0.0
        } else {
            cos_theta / (ls.distance * ls.distance)
        };
        reservoir.update(ls, weight, rng);
    }

    if let Some(sample) = reservoir.sample {
        if in_shadow(hit_point, normal, &sample, scene) {
            return Vector3::zeros();
        }
        let cos_theta = normal.dot(&sample.direction).max(0.0);
        // For Lambertian diffuse, the contribution is (light intensity * cosθ / distance²).
        // Multiplying by the surface color and dividing by pi (BRDF normalization) gives the final value.
        let contribution = sample.intensity * cos_theta / (sample.distance * sample.distance);
        return color.component_mul(&contribution) / std::f32::consts::PI;
    }
    Vector3::zeros()
}

fn main() {
    let width = 800;
    let height = 600;
    let mut img = RgbImage::new(width, height);

    // Define a simple scene: three spheres and one point light.
    let scene = Scene {
        spheres: vec![
            Sphere {
                center: Vector3::new(0.0, -0.5, -3.0),
                radius: 0.5,
                color: Vector3::new(0.7, 0.3, 0.3),
            },
            Sphere {
                center: Vector3::new(1.0, 0.0, -4.0),
                radius: 0.5,
                color: Vector3::new(0.3, 0.7, 0.3),
            },
            Sphere {
                center: Vector3::new(-1.0, 0.0, -4.0),
                radius: 0.5,
                color: Vector3::new(0.3, 0.3, 0.7),
            },
        ],
        light: Light {
            position: Vector3::new(5.0, 5.0, -2.0),
            intensity: Vector3::new(10.0, 10.0, 10.0),
        },
    };

    // Simple pinhole camera setup.
    let camera_pos = Vector3::new(0.0, 0.0, 0.0);
    let fov_deg: f32 = 90.0;
    let scale = (fov_deg.to_radians() * 0.5).tan();
    let aspect_ratio = width as f32 / height as f32;

    let mut rng = StdRng::from_entropy();

    // For each pixel, generate a ray and compute the color.
    for j in 0..height {
        for i in 0..width {
            // Convert pixel coordinate to normalized device coordinate.
            let x = (2.0 * ((i as f32 + 0.5) / width as f32) - 1.0) * aspect_ratio * scale;
            let y = (1.0 - 2.0 * ((j as f32 + 0.5) / height as f32)) * scale;
            let ray_dir = Vector3::new(x, y, -1.0).normalize();
            let ray = Ray {
                origin: camera_pos,
                direction: ray_dir,
            };

            let pixel_color =
                if let Some((_t, hit_point, normal, hit_color)) = trace_ray(&ray, &scene) {
                    shade(&hit_point, &normal, &hit_color, &scene, &mut rng)
                } else {
                    // Background color.
                    Vector3::new(0.2, 0.2, 0.2)
                };

            // Gamma correction (gamma = 2.2).
            let gamma = 2.2;
            let r = pixel_color.x.max(0.0).min(1.0).powf(1.0 / gamma);
            let g = pixel_color.y.max(0.0).min(1.0).powf(1.0 / gamma);
            let b = pixel_color.z.max(0.0).min(1.0).powf(1.0 / gamma);
            let ir = (r * 255.0) as u8;
            let ig = (g * 255.0) as u8;
            let ib = (b * 255.0) as u8;
            img.put_pixel(i, j, Rgb([ir, ig, ib]));
        }
    }

    img.save("output.png").unwrap();
}
