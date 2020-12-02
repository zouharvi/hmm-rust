r-build-time:
	@ cd rust; \
	cargo build --release --features="new_train comp_train comp_eval print_acc"
	@ # cargo build --release --features="new_train smooth"
	@ # cargo build --release --features="new_train comp_eval print_acc"

r-build-acc:
	@ cd rust; \
	cargo build --release --features="comp_train comp_eval print_acc"

r-build-smooth:
	@ cd rust; \
	cargo build --release --features="comp_train comp_eval smooth print_acc"

r-build-print-eval:
	@ cd rust; \
	cargo build --release --features="comp_eval print_pred"

r-build-print-eval-smooth:
	@ cd rust; \
	cargo build --release --features="comp_eval print_pred smooth"

r-print-eval:
	@ make r-build-print-eval
	@ ./rust/target/release/hmm-rust 1>data_measured/r-de-eval.tt
	@ make r-build-print-eval-smooth
	@ ./rust/target/release/hmm-rust 1>data_measured/r-de-eval-smooth.tt

r-run:
	@ ./rust/target/release/hmm-rust

p-run-time:
	@ python3 -O python/main.py new_train comp_train comp_eval print_acc smooth
	@ # python3 -O python/main.py new_train smooth
	@ # python3 -O python/main.py new_train comp_eval print_acc smooth

p-run-acc:
	@ python3 -O python/main.py comp_train comp_eval print_acc

p-run-smooth:
	@ python3 -O python/main.py comp_train comp_eval smooth print_acc

p-print-eval:
	@ python3 -O python/main.py comp_eval print_pred > data_measured/p-de-eval.tt
	@ python3 -O python/main.py comp_eval print_pred smooth > data_measured/p-de-eval-smooth.tt
