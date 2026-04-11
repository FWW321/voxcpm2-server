{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self
    , nixpkgs
    , fenix
    , flake-utils
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      rustToolchain = fenix.packages.${system}.complete.toolchain;

      buildDeps = with pkgs; [
        gcc
        pkg-config
        openssl
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
      ];

      cudaLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
        cudaPackages.cudatoolkit.lib
        cudaPackages.cudnn.lib
        stdenv.cc.cc.lib
      ]) + ":/run/opengl-driver/lib";

      cudaLinkPath = pkgs.lib.makeLibraryPath [
        pkgs.cudaPackages.cudatoolkit.lib
        pkgs.cudaPackages.cudnn.lib
        "/run/opengl-driver"
      ];
    in
    {
      devShells.default = pkgs.mkShell {
        packages = buildDeps ++ [ rustToolchain ];

        LD_LIBRARY_PATH = cudaLibPath;
        LIBRARY_PATH = cudaLinkPath;
        CPATH = "${pkgs.cudaPackages.cudatoolkit}/include";

        shellHook = ''
          echo "voxcpm2-rs dev shell"
          echo "rustc: $(rustc --version)"
          echo "cuda:  ${pkgs.cudaPackages.cudatoolkit.version}"
        '';
      };

      packages.default = pkgs.rustPlatform.buildRustPackage {
        pname = "voxcpm2-rs";
        version = "0.3.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;

        nativeBuildInputs = with pkgs; [ pkg-config ];
        buildInputs = with pkgs; [ openssl.out ];

        LIBRARY_PATH = cudaLinkPath;

        meta = {
          description = "VoxCPM2 TTS CLI tool (Rust)";
          mainProgram = "voxcpm2-rs";
        };
      };
    });
}
