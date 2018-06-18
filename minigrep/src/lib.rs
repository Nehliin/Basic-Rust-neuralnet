use std::fs::File;
use std::io::prelude::*;
use std::error::Error;
pub mod config;

pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let mut result = Vec::new();
    for line in contents.lines() {
        if line.contains(query) {
            result.push(line)
        }
    }
    return result;
}

pub fn run(config: config::Config) -> Result<(), Box<Error>>{
        let mut f = File::open(config.filename)?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        for line in search(&config.query, &contents) {
            println!("{}", line);
        }
        return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;
    //use::config::Config;

    #[test]
    fn searchTest() {
        let query = "duct";
        let contents = "Rust:\nsafe, fast, productive.\nPick three.";
        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }
}
