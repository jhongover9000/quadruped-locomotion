# Team robot: source files

Keep what comes out of CAD here: the URDF export in `urdf/`, its meshes in `meshes/`
(tracked with Git LFS).

`python scripts/import_robot.py assets/robots/custom/urdf/<robot>.urdf --name <name>`
turns it into the trained-on model in `assets/robots/<name>/` and the config
`configs/robots/<name>.yaml`. See [docs/robot_import.md](../../../docs/robot_import.md).
