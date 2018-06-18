pub struct Config<'a> {
    pub query : &'a String,
    pub filename :&'a String,
}

impl<'a> Config<'a> {
    pub fn new(args: &'a[String]) -> Result<Config<'a>, &'static str> {
        if args.len() < 3 {
            return Err("not enugh arguments");
        }
        let query = &args[1];
        let filename = &args[2];
        return Ok( Config {query, filename})
    }

}
