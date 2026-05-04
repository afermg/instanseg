{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              (inputs.nahual-flake.packages.${system}.nahual)
              packages.instanseg
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              export PYTHONUNBUFFERED=1
              # InstanSeg downloads weights via pkgutil + writes them next to
              # the package by default. Inside the nix store that path is
              # read-only, so redirect to a writable cache directory.
              : "''${INSTANSEG_BIOIMAGEIO_PATH:=''${XDG_CACHE_HOME:-$HOME/.cache}/instanseg/bioimageio_models}"
              export INSTANSEG_BIOIMAGEIO_PATH
              mkdir -p "$INSTANSEG_BIOIMAGEIO_PATH"
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/instanseg.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        formatter = pkgs.alejandra;

        packages = {
          instanseg = pkgs.python3.pkgs.callPackage ./nix/instanseg.nix { };
        };

        devShells = {
          default =
            let
              python_with_pkgs = (
                python3.withPackages (pp: [
                  (inputs.nahual-flake.packages.${system}.nahual)
                  packages.instanseg
                  pp.tifffile
                  pp.pyyaml
                ])
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
              '';
            };
        };
      }
    );
}
