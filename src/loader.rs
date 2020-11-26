use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::string::String;

pub struct Mapper {
    pub map_to: HashMap<String, usize>,
    pub map_from: HashMap<usize, String>,
    // points after the last one
    counter: usize,
}

impl Mapper {
    pub fn empty() -> Mapper {
        Mapper {
            map_to: HashMap::<String, usize>::new(),
            map_from: HashMap::<usize, String>::new(),
            counter: 0,
        }
    }

    pub fn update(&mut self, tok: &str) -> usize {
        if let Some(val) = self.map_to.get(tok) {
            return *val;
        } else {
            self.map_to.insert(String::from(tok), self.counter);
            self.map_from.insert(self.counter, String::from(tok));
            self.counter += 1;
            return self.counter - 1;
        }
    }

    pub fn count(&self) -> Option<usize> {
        if self.counter == 0 {
            return None;
        } else {
            return Some(self.counter - 1);
        }
    }
}

pub struct Sentence {
    pub tokens: Vec<(usize, usize)>,
}

impl Sentence {
    pub fn print(&self, mapper_w: &Mapper, mapper_t: &Mapper) {
        for x in &self.tokens {
            println!(
                "{}\t{}\t({}, {})",
                x.0,
                x.1,
                mapper_w.map_from.get(&x.0).unwrap(),
                mapper_t.map_from.get(&x.1).unwrap()
            );
        }
    }
}

pub struct Loader {
    pub data: Vec<Sentence>,
    pub mapper_w: Mapper,
    pub mapper_t: Mapper,
}

impl Loader {
    pub fn load(path: &str) -> Result<Loader, &str> {
        let mut mapper_w = Mapper::empty();
        let mut mapper_t = Mapper::empty();

        if let Ok(mut file) = File::open(path) {
            let mut contents = String::new();
            let result = file.read_to_string(&mut contents);
            if let Ok(_) = result {
                let tokens = contents.split("\n").collect::<Vec<&str>>();
                let mut data = Vec::<Sentence>::new();
                let mut sent = Sentence { tokens: vec![] };
                for line in tokens {
                    if line == "" {
                        if sent.tokens.len() != 0 {
                            data.push(sent);
                        }
                        sent = Sentence { tokens: vec![] };
                    } else {
                        let vals = line.split("\t").collect::<Vec<&str>>();
                        sent.tokens.push((
                            mapper_w.update(vals[0]),
                            if vals.len() == 1 {
                                0
                            } else {
                                mapper_t.update(vals[1])
                            },
                        ));
                    }
                }

                return Ok(Loader {
                    data,
                    mapper_w,
                    mapper_t,
                });
            }
        }

        return Err("Error loading test data");
    }

    pub fn print(&self) {
        for x in &self.data {
            x.print(&self.mapper_w, &self.mapper_t)
        }
    }
}
