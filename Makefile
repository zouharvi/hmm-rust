r-run:
	@ cd rust; \
	RUSTFLAGS=-Awarnings \
	time -f"%e" 2> ../r-time \
	cargo run --release --features="fast print_acc" 1>&2 2>/dev/null 