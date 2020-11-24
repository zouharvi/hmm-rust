use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::string::String;

pub struct Mapper {
    pub map_to: HashMap<String, usize>,
    pub map_from: HashMap<usize, String>,
}

impl Mapper {
    pub fn new(data: &Vec<Sentence>, words: bool) -> Mapper {
        let mut map_to = HashMap::<String, usize>::new();
        let mut map_from = HashMap::<usize, String>::new();

        let mut counter = 0;
        for sent in data {
            for (word, tag) in &sent.tokens {
                let token = String::from(if words { word } else { tag });
                if !map_to.contains_key(&token) {
                    map_to.insert(String::from(&token), counter);
                    map_from.insert(counter, token);
                    counter += 1;
                }
            }
        }
        return Mapper { map_to, map_from };
    }
}

pub struct Sentence {
    pub tokens: Vec<(String, String)>,
}

pub struct SentenceR {
    pub tokens: Vec<(usize, usize)>,
}

impl SentenceR {
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
    pub data_r: Vec<SentenceR>,
    pub mapper_w: Mapper,
    pub mapper_t: Mapper,
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

                let mapper_w = Mapper::new(&data, true);
                let mapper_t = Mapper::new(&data, false);

                // TODO: This can be done already at loading time, giving a performance boost
                let mut data_r = Vec::<SentenceR>::new();
                for sent in &data {
                    let mut sent_r = SentenceR { tokens: vec![] };
                    for (word, tag) in &sent.tokens {
                        let word_r = mapper_w.map_to.get(word).unwrap();
                        let tag_r = mapper_t.map_to.get(tag).unwrap();
                        sent_r.tokens.push((*word_r, *tag_r));
                    }
                    data_r.push(sent_r);
                }

                return Ok(Loader {
                    data_r,
                    mapper_w,
                    mapper_t,
                });
            }
        }

        return Err("Error loading test data");
    }

    pub fn print(&self) {
        for x in &self.data_r {
            x.print(&self.mapper_w, &self.mapper_t)
        }
    }
}
