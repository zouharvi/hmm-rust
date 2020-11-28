r-build-time:
	@ cd rust; \
	cargo build --release --features="new_train comp_dev print_acc"

r-build-acc:
	@ cd rust; \
	cargo build --release --features="comp_train comp_dev print_acc"

r-build-test:
	@ cd rust; \
	cargo build --release --features="comp_test print_pred"

r-run-test: r-build-test
	@ ./rust/target/release/hmm-rust 1>data_measured/r-de-test.tt

r-run: r-build-acc
	@ ./rust/target/release/hmm-rust

p-run-time:
	@ python3 -O python/main.py new_train comp_dev print_acc

p-run-acc:
	@ python3 -O python/main.py comp_train comp_dev print_acc

p-run-test:
	@ python3 -O python/main.py comp_test print_pred > data_measured/p-de-test.tt
