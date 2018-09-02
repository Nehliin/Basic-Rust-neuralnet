extern crate actix;
extern crate actix_redis;
extern crate actix_web;
extern crate env_logger;

use actix_redis::RedisSessionBackend;
use actix_web::middleware::session::{RequestSession, SessionStorage};
use actix_web::{middleware, client, server, App, Json, HttpRequest, HttpResponse, Result};
use actix_web::http::header;




#[derive(Serialize)]
struct MyObj {
    readOptions: ReadOption,
    keys: Vec<Key>
}

/*
readOptions": {
    object(ReadOptions)
  },
  "keys": [
    {
      object(Key)
    }
  ],

 //read options
  {

  // Union field consistency_type can be only one of the following:
  "readConsistency": enum(ReadConsistency),
  "transaction": string,
  // End of list of possible types for union field consistency_type.
}
*/
/// simple handler
fn index(req: &HttpRequest) -> Result<HttpResponse> {
    println!("{:?}", req);
    // session
    if let Some(count) = req.session().get::<i32>("counter")? {
        println!("SESSION value: {}", count);
        req.session().set("counter", count + 1)?;
        let response = format!("Welcome! SessionValue: {}", count);

        actix::run(
            || client::ClientRequest::post("https://datastore.googleapis.com/v1/projects/computetest-175618:lookup") // <- Create request builder
                .header(header::CONTENT_TYPE, "application/json")
                .finish().unwrap()
                .set_body()
                .send()                                    // <- Send http request
                .map_err(|_| ())
                .and_then(|response| {                     // <- server http response
                    println!("Response: {:?}", response);
                    Ok(())
                }),
        );
        Ok(response.into())
    } else {
        req.session().set("counter", 1)?;
        let response = format!("Welcome! SessionValue: {}", 1);
        Ok(response.into())
    }

}

fn main() {
    ::std::env::set_var("RUST_LOG", "actix_web=info,actix_redis=info");
    env_logger::init();
    let sys = actix::System::new("basic-example");

    server::new(|| {
        App::new()
            // enable logger
            .middleware(middleware::Logger::default())
            // redis session middleware
            .middleware(SessionStorage::new(
                RedisSessionBackend::new("10.0.0.3:6379", &[0; 32])
        ))
            // register simple route, handle all methods
            .resource("/", |r| r.f(index))
    }).bind("0.0.0.0:8080")
        .unwrap()
        .start();

    let _ = sys.run();
}