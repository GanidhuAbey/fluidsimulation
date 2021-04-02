use ggez::{Context, ContextBuilder, GameResult};
use ggez::event::{self, EventHandler};
use ggez::graphics;


const SIMULATION_SIZE: u32 = 40;
const FLUID_CUBE_SIZE: f32 = 10.0;

fn main() {
    // Make a Context and an EventLoop.
    let (mut ctx, mut event_loop) =
       ContextBuilder::new("game_name", "author_name")
        .window_mode(ggez::conf::WindowMode::default().dimensions(400.0, 400.0))
        .build()
        .unwrap();

    // Create an instance of your event handler.
    // Usually, you should provide it with the Context object
    // so it can load resources like images during setup.
    let mut f_sim = FluidSimulation::new(SIMULATION_SIZE);

    //fluid simulation array

    // Run!
    match event::run(&mut ctx, &mut event_loop, &mut f_sim) {
        Ok(_) => println!("Exited cleanly."),
        Err(e) => println!("Error occured: {}", e)
    }


}

struct FluidCube {
    size: u32,
    dt: f32,
    diff: f32,
    visc: f32,

    s: f32,
    density: f32,

    vx: f32,
    vy: f32,

    vx0: f32,
    vy0: f32
}

impl FluidCube {
    pub fn new(size: u32, diffusion: u32, viscosity: u32, dt: f32) -> FluidCube {
        FluidCube {
            size: size,
            dt: dt as f32,
            diff: diffusion as f32,
            visc: viscosity as f32,

            s: 0.0,
            density: 0.0,

            vx: 0.0,
            vy: 0.0,

            vx0: 0.0,
            vy0: 0.0,
        }
    }

    pub fn draw(&mut self, ctx: &mut Context, density: f32, x: u32, y: u32) -> GameResult<()> {
        let rectangle = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::fill(),
            graphics::Rect::new(x as f32 * FLUID_CUBE_SIZE, y as f32 * FLUID_CUBE_SIZE, FLUID_CUBE_SIZE, FLUID_CUBE_SIZE),
            graphics::Color::new(1.0, 0.0, 0.0, density)
        ).unwrap();
        graphics::draw(ctx, &rectangle, (ggez::mint::Point2 { x: 0.0, y: 0.0 },));
        Ok(())
    }

    pub fn add_density(&mut self, amount: f32) {
        self.density += amount;
    }

    pub fn add_velocity(&mut self, amount_x: f32, amount_y: f32) {
        self.vx += amount_x;
        self.vy += amount_y;
    }

    /*
    pub fn step_cube() {
        diffuse(1, self.vx0, self.vx, self.visc, self.dt, 4, self.size);
        diffuse(2, self.vy0, self.vy, self.visc, self.dt, 4, self.size);

        project(self.vx0, self.vy0, self.vx, self.vy, 4, self.size);

        advect(1, self.vx, self.vx0, self.vy, self.vy0, self.dt, self.N);
        advect(2, self.vx, self.vx0, self.vy, self.vy0, self.dt, self.N);

        project(self.vx, self.vy, self.vx0, self.vy0, 4, self.size);

        diffuse(0, self.s, self.density, self.diff, self.dt, 4, self.size);
        advect(0, self.density, self.s, self.vx, self.vy, self.dt, self.size);
    }
    */

}


fn set_bnd(b: u32, fluid: &mut Vec<Vec<FluidCube>>, size: usize) {
    for i in 0..SIMULATION_SIZE {
        if b == 2 {
            fluid[i as usize][0].vx = -fluid[i as usize][0].vx;
            fluid[i as usize][size - 1].vx = -fluid[i as usize][size - 1].vy;
        }
    }
    for j in 0..SIMULATION_SIZE {
        if b == 1 {
            fluid[0][j as usize].vx = -fluid[0][j as usize].vy;
            fluid[size - 1][j as usize].vx = -fluid[size - 1][j as usize].vy;
        }
    }


    //corners
    fluid[0][0].vx = 0.5 * (fluid[0][1].vx + fluid[1][0].vx);
    fluid[0][0].vy = 0.5 * (fluid[0][1].vy + fluid[1][0].vy);
    fluid[0][size-1].vx = 0.5 * (fluid[0][size-2].vx + fluid[1][size-1].vx);
    fluid[0][size-1].vy = 0.5 * (fluid[0][size-2].vy + fluid[1][size-1].vy);
    fluid[size-1][0].vx =  0.5 * (fluid[size-2][0].vx + fluid[size-1][0].vx);
    fluid[size-1][0].vy =  0.5 * (fluid[size-2][0].vy + fluid[size-1][0].vy);
    fluid[size-1][size-1].vx =  0.5 * (fluid[size-2][size-1].vx + fluid[size-1][size-2].vx);
    fluid[size-1][size-1].vy =  0.5 * (fluid[size-2][size-1].vy + fluid[size-1][size-2].vy);
}

struct FluidSimulation {
    // Stores simulation values

    //fluid vector
    fluid: Vec<Vec<FluidCube>>,

}

impl FluidSimulation {
    pub fn new(sim_size: u32) -> FluidSimulation {
        // Load/create resources here: images, fonts, sounds, etc.
        let mut fluid: Vec<Vec<FluidCube>> = Vec::new();
        for i in 0..sim_size {
            let mut row = Vec::new();
            for j in 0..sim_size {
                row.push(FluidCube::new(FLUID_CUBE_SIZE as u32, 0, 0, 0.0));
            }
            fluid.push(row);
        }

        FluidSimulation {
            fluid,
        }
    }
}

impl EventHandler for FluidSimulation {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);

        // Draw code here...
        for x in 0..SIMULATION_SIZE {
            for y in 0..SIMULATION_SIZE {
                let mut fluid_cube = &mut self.fluid[x as usize][y as usize];
                fluid_cube.draw(ctx, fluid_cube.density, x, y);
            }
        }

        graphics::present(ctx)
    }
}
