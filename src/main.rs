//use std::intrinsics::sqrtf32;
use macroquad::miniquad::window::set_window_size;
use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets, Skin};

const WINDOW_WIDTH: f32 = 1280.0;
const WINDOW_HEIGHT: f32 = 720.0;

const CENTER: Vec2 = Vec2::new(WINDOW_WIDTH / 2.0, WINDOW_HEIGHT / 2.0);

// Global Simulation Parameters
const BOUND_SIZE: f32 = 500.0;
const COLLISION_DAMPING: f32 = 0.4;


const GRAVITY: f32 = 98.0;

struct SimulationSettings {
    refresh: bool,

    gravity: f32,
    collision_damping: f32,
    bound_size: f32,
    particle_count: u32,
    particle_spacing: f32,
    particle_size: f32
}

impl SimulationSettings {
    pub fn new(gravity: f32, collision_damping: f32, bound_size: f32, particle_count: u32, particle_spacing: f32, particle_size: f32) -> Self {
        Self {
            refresh: true,
            gravity,
            collision_damping,
            bound_size,
            particle_count,
            particle_spacing,
            particle_size
        }
    }
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
        Self { radius: size, color, mass: 1.0, pos, vel: Vec2::new(0.0, 0.0) }
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
        if (offset.x > center.x) {
            self.pos.x = center.x * self.pos.signum().x;
            self.vel *= -1.0 * settings.collision_damping;
        }
        if (offset.y > center.y) {
            self.pos.y = center.y * self.pos.signum().y;
            self.vel *= -1.0 * settings.collision_damping;
        }
    }

    pub fn draw(&self) {
        draw_circle(CENTER.x + self.pos.x, CENTER.y + self.pos.y, self.radius, self.color);
    }
}

struct Fluid {
    particles: Vec<Particle>
}

impl Fluid {
    pub fn new(particle_count: u32, particle_size: f32, particle_spacing: f32) -> Self {
        let mut particles: Vec<Particle> = vec![];
        let particles_row: u32 = particle_count.isqrt();
        let row_count: u32 = particle_count / particles_row;

        let row_size = particles_row as f32 * (particle_size * 2.0 + particle_spacing);
        let col_size = row_count as f32 * (particle_size * 2.0 + particle_spacing);

        let top_left = Vec2::new(-(row_size / 2.0), -(col_size / 2.0));
        for y in 0..row_count {
            for x in 0..particles_row {
                let mut particle = Particle::new(top_left + Vec2::new(x as f32 * (2.0 * particle_size + particle_spacing), y as f32 * (2.0 * particle_size + particle_spacing)), particle_size, BLUE);
                particles.push(particle);
            }
        }

        Self { particles }
    }

    pub fn update(&mut self, delta_time: f32, settings: &SimulationSettings) {
        for particle in &mut self.particles {
            particle.update(delta_time, settings);
        }
    }

    pub fn draw(&self) {
        for particle in &self.particles {
            particle.draw();
        }
    }
}

#[macroquad::main("Basic Shapes")]
async fn main() {
    let mut global_settings = SimulationSettings::new(98.0, 0.4, 500.0, 1, 0.1, 10.0);

    // ui intialization
    let mut gravity_input: String = global_settings.gravity.to_string();
    let mut collision_damping_input: String = global_settings.collision_damping.to_string();
    let mut bound_size_input: String = global_settings.bound_size.to_string();
    let mut particle_count_input: String = global_settings.particle_count.to_string();
    let mut particle_spacing_input: String = global_settings.particle_spacing.to_string();
    let mut particle_size_input: String = global_settings.particle_size.to_string();

    set_window_size(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32);

    let mut fluid_simulation = Fluid::new(1, 0.0, 0.0);

    loop {
        clear_background(LIGHTGRAY);
        let delta_time: f32 = get_frame_time();

        draw_rectangle_lines(CENTER.x - global_settings.bound_size / 2.0,
                             CENTER.y - global_settings.bound_size / 2.0,
                             global_settings.bound_size, global_settings.bound_size, 10.0, BLACK);

        // restart fluid simulation.
        if (global_settings.refresh) {
            global_settings.refresh = false;

            fluid_simulation = Fluid::new(global_settings.particle_count, global_settings.particle_size, global_settings.particle_spacing);
        }

        // UI to manage parameters
        root_ui().group(hash!(), vec2(250.0, 180.0), |ui| {
            ui.label(None, "Simulation Settings");

            ui.input_text(hash!(), "Gravity", &mut gravity_input);
            ui.input_text(hash!(), "Collision Damping", &mut collision_damping_input);
            ui.input_text(hash!(), "Bounds", &mut bound_size_input);
            ui.input_text(hash!(), "Particle Count", &mut particle_count_input);
            ui.input_text(hash!(), "Particle Spacing", &mut particle_spacing_input);
            ui.input_text(hash!(), "Particle Size", &mut particle_size_input);

            if ui.button(None, "Apply") {
                // cast to float
                global_settings.gravity = gravity_input.parse::<f32>().unwrap();
                global_settings.collision_damping = collision_damping_input.parse::<f32>().unwrap();
                global_settings.bound_size = bound_size_input.parse::<f32>().unwrap();
                global_settings.particle_count = particle_count_input.parse::<u32>().unwrap();
                global_settings.particle_spacing = particle_spacing_input.parse::<f32>().unwrap();
                global_settings.particle_size = particle_size_input.parse::<f32>().unwrap();
                global_settings.refresh = true;
            }
        });

        fluid_simulation.update(delta_time, &global_settings);
        fluid_simulation.draw();

        next_frame().await;
    }
}
