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
        llvmPackages.libclang
        ffmpeg.dev
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
      ];

      runtimeDeps = with pkgs; [
        openssl.out
        ffmpeg
      ];

      cudaLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
        cudaPackages.cudatoolkit.lib
        cudaPackages.cudnn.lib
        stdenv.cc.cc.lib
        ffmpeg.lib
      ]) + ":/run/opengl-driver/lib";

      cudaLinkPath = pkgs.lib.makeLibraryPath [
        pkgs.cudaPackages.cudatoolkit.lib
        pkgs.cudaPackages.cudnn.lib
        "/run/opengl-driver"
        pkgs.ffmpeg.lib
      ];
    in
    {
      devShells.default = pkgs.mkShell {
        packages = buildDeps ++ [ rustToolchain ];

        LD_LIBRARY_PATH = cudaLibPath;
        LIBRARY_PATH = cudaLinkPath;
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        CPATH = "${pkgs.cudaPackages.cudatoolkit}/include:${pkgs.ffmpeg.dev}/include";
        BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.glibc.dev}/include";

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
    })
    // {
      nixosModule = { config, lib, pkgs, ... }:
    let
      cfg = config.services.voxcpm2-server;
    in
    {
      options.services.voxcpm2-server = {
        enable = lib.mkEnableOption "VoxCPM2 TTS inference server";

        package = lib.mkOption {
          type = lib.types.package;
          description = "VoxCPM2 server package";
        };

        host = lib.mkOption {
          type = lib.types.str;
          default = "0.0.0.0";
          description = "Host to bind";
        };

        port = lib.mkOption {
          type = lib.types.port;
          default = 5800;
          description = "Port to bind";
        };

        modelDir = lib.mkOption {
          type = lib.types.path;
          default = "/var/lib/voxcpm2-server/model";
          description = "Path to VoxCPM2 model directory";
        };

        cudaLibPath = lib.mkOption {
          type = lib.types.str;
          description = "LD_LIBRARY_PATH for CUDA libraries";
        };

        user = lib.mkOption {
          type = lib.types.str;
          default = "voxcpm2-server";
          description = "User to run the service as";
        };

        group = lib.mkOption {
          type = lib.types.str;
          default = "voxcpm2-server";
          description = "Group to run the service as";
        };
      };

      config = lib.mkIf cfg.enable {
        users.users.${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = "/var/lib/voxcpm2-server";
          createHome = true;
        };
        users.groups.${cfg.group} = {};

        systemd.services.voxcpm2-server = {
          wantedBy = [ "multi-user.target" ];
          after = [ "network.target" ];
          description = "VoxCPM2 TTS inference server";

          serviceConfig = {
            Type = "simple";
            ExecStart = "${lib.getExe cfg.package} --model ${cfg.modelDir} --host ${cfg.host} --port ${toString cfg.port}";
            Restart = "on-failure";
            RestartSec = 5;
            User = cfg.user;
            Group = cfg.group;
            Environment = lib.optionalString (cfg.cudaLibPath != "") "LD_LIBRARY_PATH=${cfg.cudaLibPath}";
            WorkingDirectory = "/var/lib/voxcpm2-server";
          };
        };
      };
    };
  };
}
