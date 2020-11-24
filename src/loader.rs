use std::fs::File;
use std::io::prelude::*;
use std::string::String;

pub struct Sentence {
    tokens: Vec<(String, String)>,
}

impl Sentence {
    pub fn print(self) {
        for x in self.tokens {
            println!("{}\t{}", x.0, x.1);
        }
    }
}

pub struct Loader {
    data: Vec<Sentence>,
}

impl Loader {
    pub fn load(path: &str) -> Result<Loader, &str> {
        if let Ok(mut file) = File::open(path) {
            let mut contents = String::new();
            let result = file.read_to_string(&mut contents);
            if let Ok(_) = result {
                let tokens = contents.split("\n").collect::<Vec<&str>>();
                let mut data = Vec::<Sentence>::new();
                let mut sent = Sentence { tokens: vec![] };
                for line in tokens {
                    if line == "" {
                        data.push(sent);
                        sent = Sentence { tokens: vec![] };
                    } else {
                        let vals = line.split("\t").collect::<Vec<&str>>();
                        sent.tokens.push((
                            String::from(vals[0]),
                            if vals.len() == 1 {
                                String::from("")
                            } else {
                                String::from(vals[1])
                            },
                        ));
                    }
                }

                return Ok(Loader { data });
            }
        }

        return Err("Error loading test data");
    }

    pub fn print(self) {
        for x in self.data {
            x.print()
        }
    }
}
