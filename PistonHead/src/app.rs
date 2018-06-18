extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

use app::piston::window::WindowSettings;
use app::piston::event_loop::*;
use app::piston::input::*;
use app::glutin_window::GlutinWindow as Window;
use app::opengl_graphics::{GlGraphics, OpenGL };

struct Position{
    x : f64,
    y : f64,
    vx: f64,
    vy: f64,
    pressedXPositive : bool,
    pressedXNegative : bool,
    pressedYPositive : bool,
    pressedYNegative : bool,
    XMAX : u32,
    XMIN : u32,
    YMAX : u32,
    YMIN : u32,
}

pub struct App {
    pos : Position,
    speed: f64,
    window: Window,
    events : Events,
    gl : GlGraphics
}

impl App {

    pub fn new(width: u32, height : u32) -> App {
        // Change this to OpenGL::V2_1 if not working
        let opengl = OpenGL::V3_2;
        // Create an Glutin window.
        let window: Window = WindowSettings::new(
            "glace blazwe",
            [width, height]
        )
            .opengl(opengl)
            .exit_on_esc(true)
            .build()
            .unwrap();
        let events = Events::new(EventSettings::new());

        return App {
            pos : Position {
                x : 0.0,
                y : 0.0,
                vx: 2.0,
                vy: 2.0,
                pressedXPositive : false,
                pressedXNegative : false,
                pressedYPositive : false,
                pressedYNegative : false,
                XMAX : width,
                XMIN : 0,
                YMAX : height,
                YMIN : 0
            },
            speed : 2.0,
            window,
            events,
            gl: GlGraphics::new(opengl)
        };
    }

    pub fn run(&mut self) {

        while let Some(e) = self.events.next(&mut self.window) {

            e.press(| key : piston::input::Button | {
                match key {
                    Button::Keyboard(Key::A) => self.pos.pressedXNegative = true,
                    Button::Keyboard(Key::D) => self.pos.pressedXPositive = true,
                    Button::Keyboard(Key::W) => self.pos.pressedYNegative = true,
                    Button::Keyboard(Key::S) => self.pos.pressedYPositive = true,
                    _ => (),
                }
            });
            e.release(| key : piston::input::Button | {
                match key {
                    Button::Keyboard(Key::A) => self.pos.pressedXNegative = false,
                    Button::Keyboard(Key::D) => self.pos.pressedXPositive = false,
                    Button::Keyboard(Key::W) => self.pos.pressedYNegative = false,
                    Button::Keyboard(Key::S) => self.pos.pressedYPositive = false,
                    _ => (),
                }
            });


            /*if let Some() = e.release(| key : piston::input::Button | {
                match key {
                    Button::Keyboard(Key::A) => self.pos.pressedX = false,
                    Button::Keyboard(Key::D) => self.pos.vx += self.speed,
                    Button::Keyboard(Key::W) => self.pos.vy -= self.speed,
                    Button::Keyboard(Key::S) => self.pos.vy += self.speed,
                    _ => (),
                }
            }) {
            }*/

            if let Some(r) = e.render_args() {
                self.render(&r);
            }

            if let Some(u) = e.update_args() {
                self.update(&u);
            }

        }
    }


    fn render(&mut self, args: &RenderArgs) {
        use self::graphics::*;
        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const RED:   [f32; 4] = [1.0, 0.0, 0.0, 1.0];

        let square = rectangle::square((args.width/2) as f64, (args.height/2) as f64, 50.0);

        let x = self.pos.x;
        let y = self.pos.y;
        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(GREEN, gl);

            let transform = c.transform.trans(x, y);

            // Draw a box rotating around the middle of the screen.
            rectangle(RED, square, transform, gl);
        });
    }

    fn update(&mut self, args: &UpdateArgs) {

        // Rotate 2 radians per second.
        if self.pos.pressedXPositive {
            println!("x pos: {}", self.pos.x);
            if self.pos.x < self.pos.XMAX as f64 {
                self.pos.x += self.pos.vx;
            }
        }

        if self.pos.pressedXNegative {
            println!("x pos: {}", self.pos.x);
            if self.pos.x > self.pos.XMIN as f64 {
                self.pos.x -= self.pos.vx;
            }
        }

        if self.pos.pressedYPositive {
            println!("y pos: {}", self.pos.y);
            if self.pos.y < self.pos.YMAX as f64 {
                self.pos.y += self.pos.vy;
            }
        }

        if self.pos.pressedYNegative {
            println!("y pos: {}", self.pos.y);
            if self.pos.y > self.pos.YMIN as f64 {
                self.pos.y -= self.pos.vy;
            }
        }
    }

}
