r-build-time:
	@ cd rust; \
	cargo build --release --features="new_train print_acc"

r-build-acc:
	@ cd rust; \
	cargo build --release --features="comp_train comp_dev print_acc"

r-build-test:
	@ cd rust; \
	cargo build --release --features="comp_test print_pred"

r-run-out: r-build-time
	@ time -a -f"%e" 2>data_measured/r-time \
	./rust/target/release/hmm-rust 1>data_measured/r-out

r-run-acc: r-build-acc
	@ ./rust/target/release/hmm-rust

p-run-time:
	@ python3 python/main.py new_train print_acc

p-run-acc:
	@ python3 python/main.py comp_train comp_dev print_acc

p-run-test:
	@ python3 python/main.py comp_test print_pred > data_measured/p-out