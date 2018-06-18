

mod app;

fn main() {
    use app::App;

    // Create a new game and run it.
    let mut app = App::new(500, 500);
    app.run();



}