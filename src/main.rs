// This project was inspired by Sebastian Lague
// His Video: https://www.youtube.com/watch?v=rSKMYc1CQHE&t=6s

use std::cmp::max;
use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation};
use macroquad::miniquad::window::set_window_size;
use macroquad::prelude::*;
use macroquad::rand::{gen_range};
use macroquad::ui::{hash, root_ui, InputHandler};

const WINDOW_WIDTH: f32 = 600.0;
const WINDOW_HEIGHT: f32 = 600.0;

const PI: f32 = 3.14159;

const MAX_PARTICLE_SIZE: f32 = 10.0;

const CENTER: Vec2 = Vec2::new(WINDOW_WIDTH / 2.0, WINDOW_HEIGHT / 2.0);

// Global Simulation Parameters
struct SimulationSettings {
    refresh: bool,
    random_spawn: bool,

    gravity: f32,
    collision_damping: f32,
    bound_size: f32,
    particle_count: u32,
    particle_spacing: f32,

    target_density: f32,
    pressure_multiplier: f32,

    // particle size controls the range of influence that particle has.
    particle_influence: f32
}

// Helper functions
fn inverse_lerp(a: f32, b: f32, s: f32) -> f32 {
    // s -> (a, b)
    // s - a -> (0, b - a)
    // (s - a)/(b - a) -> (0, 1)

    (s - a) / (b - a)
}

fn density_to_pressure(density: f32, target: f32, pressure_multiplier: f32) -> f32 {
    let error = density - target;
    let pressure = error * pressure_multiplier;
    pressure
}

fn smoothing_kernel(r: f32, d: f32) -> f32 {
    let value = (0.0_f32).max(r - d);

    // TODO: this volume calculation may be incorrect...
    let volume = PI * r.powi(4) / 2.0;

    value * value * value / volume
}

fn smoothing_kernel_gradient(r: f32, d: f32) -> f32 {
    // df/dx = 3(r - (x^2 + y^2)^(1/2))^2*((1/2)(x^2+y^2)^(-1/2))*2x
    // df/dx = 6x/2sqrt(x^2 + y^2)*(r-sqrt(x^2+y^2))^2
    // df/dx = 6x/2d*(r-d)^2
    // df/dy = 6y/2d*(r-d)^2
    // gradient = -6*d*f^2

    //if d.length() >= r { return Vec2::new(0.0, 0.0); }
    //let f = r - d.length();
    let volume = PI * r.powi(4) / 2.0;

    //Vec2::new(-6.0 * d.x * 2.0 * d.length() * f * f / volume, -6.0 * d.y * d.length() * f * f / volume)

    if d >= r { return 0.0; }
    let f = r - d;

    -3.0*f*f/volume
}

impl SimulationSettings {
    pub fn new(random_spawn: bool,
               gravity: f32,
               collision_damping: f32,
               bound_size: f32,
               particle_count: u32,
               particle_spacing: f32,
               target_density: f32,
               pressure_multiplier: f32,
               particle_size: f32) -> Self {
        Self {
            refresh: true,
            random_spawn,
            gravity,
            collision_damping,
            bound_size,
            particle_count,
            particle_spacing,
            target_density,
            pressure_multiplier,
            particle_influence: particle_size
        }
    }
}

fn create_particle_material() -> Material {
    let pipeline_params = PipelineParams {
        color_blend: Some(BlendState::new(
            Equation::Add,
            BlendFactor::Value(BlendValue::SourceAlpha),
            BlendFactor::OneMinusValue(BlendValue::SourceAlpha))
        ),
        ..Default::default()
    };

    load_material(
        ShaderSource::Glsl {
            vertex: PARTICLE_VERTEX,
            fragment: PARTICLE_FRAG
        },
        MaterialParams {
            pipeline_params,
            uniforms: vec![UniformDesc::new("EndPoint", UniformType::Float2),
                           UniformDesc::new("Center", UniformType::Float2)],
            ..Default::default()
        }
    ).unwrap()
}

struct Particle {
    radius: f32,
    color: Color,
    mass: f32,

    pos: Vec2,
    vel: Vec2,
}

impl Particle {
    pub fn new(pos: Vec2, size: f32, color: Color) -> Self {
        Self { radius: size, color, mass: 1.0, pos, vel: Vec2::new(0.0, 0.0)}
    }

    pub fn update(&mut self, delta_time: f32, settings: &SimulationSettings) {
        // move fluid under gravity
        self.vel += Vec2::new(0.0, 1.0) * delta_time * settings.gravity;
        self.pos += self.vel * delta_time;

        // resolve collisions
        self.collide(settings);
    }

    fn collide(&mut self, settings: &SimulationSettings) {
        let center = Vec2::new(settings.bound_size / 2.0, settings.bound_size / 2.0) - Vec2::new(1.0, 1.0) * self.radius;
        let offset = self.pos.abs();
        if offset.x > center.x {
            self.pos.x = center.x * self.pos.signum().x;
            self.vel *= -1.0 * settings.collision_damping;
        }
        if offset.y > center.y {
            self.pos.y = center.y * self.pos.signum().y;
            self.vel *= -1.0 * settings.collision_damping;
        }
    }

    pub fn draw(&self, material: &mut Material) {
        let center = CENTER + self.pos;
        material.set_uniform("Center", center);
        material.set_uniform("EndPoint", Vec2::new(center.x + self.radius, center.y));
        gl_use_material(material);
        draw_circle(center.x, center.y, self.radius, self.color);
        gl_use_default_material();
        draw_circle(center.x, center.y, self.radius / 10.0, BLACK);
    }
}

struct Fluid {
    particles: Vec<Particle>,
    densities: Vec<f32>
}

impl Fluid {
    pub fn new_random(particle_count: u32, particle_size: f32, bound_size: f32) -> Self {
        let mut particles: Vec<Particle> = Vec::with_capacity(particle_count as usize);
        let mut densities: Vec<f32> = Vec::with_capacity(particle_count as usize);
        let mut particle_color = BLUE;
        particle_color.a = 1.0 - clamp(inverse_lerp(1.0, MAX_PARTICLE_SIZE, particle_size), 0.0, 1.0);

        for _i in 0..particle_count {
            let x = gen_range(-bound_size / 2.0, bound_size / 2.0);
            let y = gen_range(-bound_size / 2.0, bound_size / 2.0);
            particles.push(Particle::new(Vec2::new(x, y), particle_size, particle_color));
            densities.push(0.0);
        }

        Self { particles, densities }
    }

    // - - - - - -
    // - - - - - -
    // - - - - - -
    // - - - - - -
    // - - - - - -
    pub fn new(particle_count: u32, particle_size: f32, particle_spacing: f32) -> Self {
        let mut particles: Vec<Particle> = vec![];
        let mut densities: Vec<f32> = vec![];
        let mut particle_color = BLUE;
        particle_color.a = 1.0 - clamp(inverse_lerp(1.0, MAX_PARTICLE_SIZE, particle_size), 0.0, 1.0);

        let particles_row: u32 = particle_count.isqrt();
        let particles_col: u32 = particle_count / particles_row;

        let row_size = particles_row as f32 * (particle_size * 2.0 + particle_spacing);
        let col_size = particles_col as f32 * (particle_size * 2.0 + particle_spacing);

        let top_left = Vec2::new(-(row_size / 2.0), -(col_size / 2.0));
        for y in 0..particles_col {
            for x in 0..particles_row {
                let particle = Particle::new(Vec2::new(x as f32 * particle_spacing,
                                         y as f32 * particle_spacing),
                    particle_size,
                    particle_color);
                particles.push(particle);
                densities.push(0.0);
            }
        }

        Self { particles, densities }
    }

    pub fn calculate_pressure_force(&mut self, particle_index: usize, settings: &SimulationSettings) -> Vec2 {
        let mut pressure_force = Vec2::new(0.0, 0.0);
        for i in 0..self.particles.len() {
            if particle_index == i {continue;}
            // compute gradient (direction of change) based on
            let mut direction = self.particles[particle_index].pos - self.particles[i].pos;
            if (direction.length() == 0.0) {
                // select random direction
                direction = Vec2::new(gen_range(-1.0, 1.0), gen_range(-1.0, 1.0));;
            }
            let slope = smoothing_kernel_gradient(self.particles[i].radius, direction.length());
            direction = direction.normalize();
            let density = self.densities[i];
            let pressure = density_to_pressure(density, settings.target_density, settings.pressure_multiplier);
            pressure_force += -pressure * direction * slope / density;
        }

        pressure_force
    }
    pub fn compute_densities(&mut self, settings: &SimulationSettings) {
        for i in 0..self.densities.len() {
            self.densities[i] = self.compute_density(settings, self.particles[i].pos);
        }
    }
    pub fn compute_density(&self, settings: &SimulationSettings, sample_point: Vec2) -> f32 {
        let mut density = 0.0;

        // convert sample_point from pixel space to world space.
        let half_bound_size = Vec2::new(1.0, 1.0) * settings.bound_size / 2.0;

        for particle in &self.particles {
            let difference = (sample_point - particle.pos).abs();
            density += smoothing_kernel(particle.radius, difference.length()) * particle.mass;
        }
        density
    }

    pub fn update(&mut self, delta_time: f32, settings: &SimulationSettings) {
        for particle in &mut self.particles {
            // move fluid under gravity
            particle.vel += Vec2::new(0.0, 1.0) * delta_time * settings.gravity;
            particle.pos += particle.vel * delta_time;

        }

        for i in 0..self.particles.len() {
            let pressure_force = self.calculate_pressure_force(i, settings);
            let acceleration = pressure_force / self.densities[i];

            self.particles[i].vel += acceleration * delta_time;
        }

        for particle in &mut self.particles {
            // resolve collisions
            particle.collide(settings);
        }
    }

    pub fn draw(&self, material: &mut Material) {
        for particle in &self.particles {
            particle.draw(material);
        }
    }
}

#[macroquad::main("Basic Shapes")]
async fn main() {
    let mut global_settings = SimulationSettings::new(false,0.0, 0.4, 500.0, 1, 0.1, 0.0, 0.0, 10.0);

    let mut material = create_particle_material();

    // ui intialization
    let mut gravity_input: String = global_settings.gravity.to_string();
    let mut collision_damping_input: String = global_settings.collision_damping.to_string();
    let mut bound_size_input: String = global_settings.bound_size.to_string();
    let mut particle_count_input: String = global_settings.particle_count.to_string();
    let mut particle_spacing_input: String = global_settings.particle_spacing.to_string();
    let mut particle_influence_input: String = global_settings.particle_influence.to_string();
    let mut target_density_input: String = global_settings.target_density.to_string();
    let mut pressure_multiplier_input: String = global_settings.pressure_multiplier.to_string();

    set_window_size(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32);

    let mut fluid_simulation = Fluid::new(1, 0.0, 0.0);
    fluid_simulation.compute_densities(&global_settings);

    loop {
        clear_background(LIGHTGRAY);
        let delta_time: f32 = get_frame_time();


        if is_mouse_button_pressed(MouseButton::Left) {
            fluid_simulation.compute_density(&global_settings, Vec2::new(mouse_position().0, mouse_position().1));
        }

        draw_rectangle_lines(CENTER.x - global_settings.bound_size / 2.0,
                             CENTER.y - global_settings.bound_size / 2.0,
                             global_settings.bound_size, global_settings.bound_size, 10.0, BLACK);

        // restart fluid simulation.
        if global_settings.refresh {
            global_settings.refresh = false;

            fluid_simulation = if global_settings.random_spawn { Fluid::new_random(global_settings.particle_count, global_settings.particle_influence, global_settings.bound_size) }
                                else { Fluid::new(global_settings.particle_count, global_settings.particle_influence, global_settings.particle_spacing) };
            fluid_simulation.compute_densities(&global_settings);
        }

        // UI to manage parameters
        root_ui().group(hash!(), vec2(250.0, 200.0), |ui| {
            ui.label(None, "Simulation Settings");

            ui.checkbox(hash!(), "Random Spawn", &mut global_settings.random_spawn);

            ui.input_text(hash!(), "Gravity", &mut gravity_input);
            ui.input_text(hash!(), "Collision Damping", &mut collision_damping_input);
            ui.input_text(hash!(), "Bounds", &mut bound_size_input);
            ui.input_text(hash!(), "Particle Count", &mut particle_count_input);
            ui.input_text(hash!(), "Particle Spacing", &mut particle_spacing_input);
            ui.input_text(hash!(), "Particle Influence", &mut particle_influence_input);
            ui.input_text(hash!(), "Target Density", &mut target_density_input);
            ui.input_text(hash!(), "Pressure Multiplier", &mut pressure_multiplier_input);

            if ui.button(None, "Apply") {
                // cast to float
                global_settings.gravity = gravity_input.parse::<f32>().unwrap();
                global_settings.collision_damping = collision_damping_input.parse::<f32>().unwrap();
                global_settings.bound_size = bound_size_input.parse::<f32>().unwrap();
                global_settings.particle_count = particle_count_input.parse::<u32>().unwrap();
                global_settings.particle_spacing = particle_spacing_input.parse::<f32>().unwrap();
                global_settings.particle_influence = particle_influence_input.parse::<f32>().unwrap();
                global_settings.pressure_multiplier = pressure_multiplier_input.parse::<f32>().unwrap();
                global_settings.target_density = target_density_input.parse::<f32>().unwrap();
                global_settings.random_spawn = global_settings.random_spawn;
                global_settings.refresh = true;
            }
        });

        fluid_simulation.update(delta_time, &global_settings);
        fluid_simulation.draw(&mut material);

        fluid_simulation.compute_densities(&global_settings);

        next_frame().await;
    }
}

/* TODO : move to separate file */
// Shader sources
const PARTICLE_FRAG: &'static str = r#"#version 100
precision lowp float;

varying vec2 uv;
varying vec2 uv_screen;
varying vec2 center;
varying float radius;

// distance - normalized param
float smoothing_function(float d, float r) {
    float value = max(0.0, (r * r - d * d));
    return (15.0 / (16.0 * 3.14159 * pow(r, 5.0))) * value * value;
}

void main() {
    float epsilon = 0.1;
    float alpha = smoothing_function(length(uv_screen - center), radius);
    vec3 color = vec3(0.0, 0.47, 0.95);

    gl_FragColor = vec4(color, alpha);
}
"#;

const PARTICLE_VERTEX: &'static str = "#version 100
attribute vec3 position;

varying lowp float radius;
varying lowp vec2 center;
varying lowp vec2 uv;
varying lowp vec2 uv_screen;

uniform mat4 Model;
uniform mat4 Projection;

uniform vec2 Center;
uniform vec2 EndPoint;

void main() {
    // TODO: Theres is a clear bug in this projection matrix which doesn't
    //       account for the aspect ratio of the window...
    vec4 res = Projection * Model * vec4(position, 1);
    vec4 c = Projection * Model * vec4(Center, 0, 1);
    vec4 r = Projection * Model * vec4(EndPoint, 0, 1);

    uv_screen = res.xy / 2.0 + vec2(0.5, 0.5);
    center = (c.xy + vec2(1.0, 1.0)) / 2.0;
    radius = length((r.xy / 2.0 + vec2(0.5, 0.5)) - center);

    gl_Position = res;
}
";