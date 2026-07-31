{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        instanseg = pkgs.python3.pkgs.callPackage ./nix/instanseg.nix {};
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
          instanseg
        ]);
        runServer = pkgs.writeScriptBin "nahual-instanseg" ''
          #!${pkgs.bash}/bin/bash
          export PYTHONSAFEPATH=1
          : "''${INSTANSEG_BIOIMAGEIO_PATH:=''${XDG_CACHE_HOME:-$HOME/.cache}/instanseg/bioimageio_models}"
          export INSTANSEG_BIOIMAGEIO_PATH
          mkdir -p "$INSTANSEG_BIOIMAGEIO_PATH"
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        instansegApp = {
          type = "app";
          program = "${runServer}/bin/nahual-instanseg";
        };
      in
        with pkgs; rec {
          packages =
            {inherit instanseg;}
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "instanseg";
                title = "Nahual InstanSeg";
                description = "InstanSeg instance segmentation served through Nahual";
                source = "https://github.com/afermg/instanseg";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = instansegApp.program;
              };
            };
          inherit python_with_pkgs;
          formatter = pkgs.alejandra;
          scripts.runServer = runServer;
          apps = rec {
            instanseg = instansegApp;
            default = instanseg;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              python3Packages.tifffile
              python3Packages.pyyaml
            ];
            shellHook = ''
              export PYTHONSAFEPATH=1
              : "''${INSTANSEG_BIOIMAGEIO_PATH:=''${XDG_CACHE_HOME:-$HOME/.cache}/instanseg/bioimageio_models}"
              export INSTANSEG_BIOIMAGEIO_PATH
            '';
          };
        }
    );
}
