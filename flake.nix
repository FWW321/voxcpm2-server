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

      runtimeDeps = with pkgs; [
        openssl.out
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
          echo "voxcpm2-server dev shell"
          echo "rustc: $(rustc --version)"
          echo "cuda:  ${pkgs.cudaPackages.cudatoolkit.version}"
        '';
      };

      packages.default = pkgs.rustPlatform.buildRustPackage {
        pname = "voxcpm2-server";
        version = "0.1.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;

        buildInputs = runtimeDeps;
        nativeBuildInputs = with pkgs; [ pkg-config ];

        LIBRARY_PATH = cudaLibPath;

        meta = {
          description = "VoxCPM2 TTS inference server";
          mainProgram = "voxcpm2-server";
        };
      };
    });
}
