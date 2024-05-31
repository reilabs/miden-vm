{
  description = "Miden-Gpu";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        pkgsAllowUnfree = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        gcc12 = pkgs.overrideCC pkgs.stdenv pkgs.gcc12;
        nightlyToolchain = pkgs.rust-bin.selectLatestNightlyWith (toolchain:
          toolchain.minimal.override { extensions = [ "rustfmt" ]; });

        stableToolchain = pkgs.rust-bin.stable.latest.minimal.override {
          extensions = [ "clippy" "llvm-tools-preview" "rust-src" ];
          targets = [ "x86_64-pc-windows-msvc" "x86_64-unknown-linux-gnu" "aarch64-apple-darwin" ];
        };
        # A script that calls nightly cargo if invoked with `+nightly`
        cargo-with-nightly = pkgs.writeShellScriptBin "cargo" ''
          if [[ "$1" == "+nightly" ]]; then
            shift
            # Prepend nightly toolchain directory containing cargo, rustc, etc.
            exec env PATH="${nightlyToolchain}/bin:$PATH" cargo "$@"
          fi
          exec ${stableToolchain}/bin/cargo "$@"
        '';
        baseShell = with pkgs;
          clang16Stdenv.mkDerivation {
            name = "base-shell";
            buildInputs = [
              pkg-config
              git

              cargo-with-nightly
              stableToolchain
              nightlyToolchain
              cargo-sort

              clang-tools_16
              clangStdenv
              llvm_16
            ] ++ lib.optionals stdenv.isDarwin
              [  ];

            CARGO_TARGET_DIR = "target/nix_rustc";

            shellHook = ''
              export RUST_BACKTRACE=full
              export PATH="$PATH:$(pwd)/target/debug:$(pwd)/target/release"
              # Prevent cargo aliases from using programs in `~/.cargo` to avoid conflicts with local rustup installations.
              export CARGO_HOME=$HOME/.cargo-nix

              # Ensure `cargo fmt` uses `rustfmt` from nightly.
              export RUSTFMT="${nightlyToolchain}/bin/rustfmt"

              export C_INCLUDE_PATH="${llvmPackages_16.libclang.lib}/lib/clang/${llvmPackages_16.libclang.version}/include"
              export LIBCLANG_PATH=
              export CC="${clang-tools_16.clang}/bin/clang"
              export CXX="${clang-tools_16.clang}/bin/clang++"
              export AR="${llvm_16}/bin/llvm-ar"
              export CFLAGS="-mcpu=generic"
            '';
          };
      in
      with pkgs; {
        devShell = baseShell;
        devShells = {
          cudaShell =
            let cudatoolkit = pkgsAllowUnfree.cudaPackages_12_3.cudatoolkit;
            in baseShell.overrideAttrs (oldAttrs: {
              name = "cuda-shell";
              buildInputs = oldAttrs.buildInputs
                ++ [
                  cmake
                  cudatoolkit
                  gcc12
                  pkgsAllowUnfree.cudaPackages.cuda_nvcc
                ];
              shellHook = oldAttrs.shellHook + ''
                export PATH="${pkgs.gcc12}/bin:${cudatoolkit}/bin:${cudatoolkit}/nvvm/bin:$PATH"
                export LD_LIBRARY_PATH=${cudatoolkit}/lib
                export CUDA_PATH=${cudatoolkit}
                export CPATH="${cudatoolkit}/include"
                export LIBRARY_PATH="$LIBRARY_PATH:/lib"
                export CMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc
                export CFLAGS=""
              '';
            });
        };
      });
}