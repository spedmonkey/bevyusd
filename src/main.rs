use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

// Define a "marker" component to mark the custom mesh. Marker components are often used in Bevy for
// filtering entities in queries with `With`, they're usually not queried directly since they don't
// contain information within them.
#[derive(Component)]
struct CustomUV;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, input_handler)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // Import the custom texture.
    let custom_texture_handle: Handle<Image> = asset_server.load("textures/test.png");
    // Create and save a handle to the mesh.
    let cube_mesh_handle: Handle<Mesh> = meshes.add(create_cube_mesh());

    // Render the mesh with the custom texture, and add the marker.
    commands.spawn((
        Mesh3d(cube_mesh_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(custom_texture_handle),
            ..default()
        })),
        CustomUV,
    ));

    // Transform for the camera and lighting, looking at (0,0,0) (the position of the mesh).
    let camera_and_light_transform =
        Transform::from_xyz(1.8, 1.8, 1.8).looking_at(Vec3::ZERO, Vec3::Y);

    // Camera in 3D space.
    commands.spawn((Camera3d::default(), camera_and_light_transform));

    // Light up the scene.
    commands.spawn((PointLight::default(), camera_and_light_transform));

    // Text to describe the controls.
    commands.spawn((
        Text::new("Controls:\nSpace: Change UVs\nX/Y/Z: Rotate\nR: Reset orientation"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

// System to receive input from the user,
// check out examples/input/ for more examples about user input.
fn input_handler(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mesh_query: Query<&Mesh3d, With<CustomUV>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<&mut Transform, With<CustomUV>>,
    time: Res<Time>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        let mesh_handle = mesh_query.single().expect("Query not successful");
        let mesh = meshes.get_mut(mesh_handle).unwrap();
        toggle_texture(mesh);
    }
    if keyboard_input.pressed(KeyCode::KeyX) {
        for mut transform in &mut query {
            transform.rotate_x(time.delta_secs() / 1.2);
        }
    }
    if keyboard_input.pressed(KeyCode::KeyY) {
        for mut transform in &mut query {
            transform.rotate_y(time.delta_secs() / 1.2);
        }
    }
    if keyboard_input.pressed(KeyCode::KeyZ) {
        for mut transform in &mut query {
            transform.rotate_z(time.delta_secs() / 1.2);
        }
    }
    if keyboard_input.pressed(KeyCode::KeyR) {
        for mut transform in &mut query {
            transform.look_to(Vec3::NEG_Z, Vec3::Y);
        }
    }
}

#[rustfmt::skip]
fn create_cube_mesh() -> Mesh {
    // Keep the mesh data accessible in future frames to be able to mutate it in toggle_texture.
    //println!("{:?}", get_verts());

    
    let points =         vec![
            // top (facing towards +y)
            [-0.5, 0.5, -0.5], // vertex with index 0
            [0.5, 0.5, -0.5], // vertex with index 1
            [0.5, 0.5, 0.5], // etc. until 23
            [-0.5, 0.5, 0.5],
            // bottom   (-y)
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            // right    (+x)
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5], // This vertex is at the same position as vertex with index 2, but they'll have different UV and normal
            [0.5, 0.5, -0.5],
            // left     (-x)
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            // back     (+z)
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            // forward  (-z)
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ];

    let points_b =         vec![
            // top (facing towards +y)
            vec![-0.5, 0.5, -0.5], // vertex with index 0
            vec![0.5, 0.5, -0.5], // vertex with index 1
            vec![0.5, 0.5, 0.5], // etc. until 23
            vec![-0.5, 0.5, 0.5],
            vec![-0.5, -0.5, -0.5],
            vec![0.5, -0.5, -0.5],
            vec![0.5, -0.5, 0.5],
            vec![-0.5, -0.5, 0.5],
            vec![0.5, -0.5, -0.5],
            vec![0.5, -0.5, 0.5],
            vec![0.5, 0.5, 0.5], // This vertex is at the same position as vertex with index 2, but they'll have different UV and normal
            vec![0.5, 0.5, -0.5],
            vec![-0.5, -0.5, -0.5],
            vec![-0.5, -0.5, 0.5],
            vec![-0.5, 0.5, 0.5],
            vec![-0.5, 0.5, -0.5],
            vec![-0.5, -0.5, 0.5],
            vec![-0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5],
            vec![0.5, -0.5, 0.5],
            vec![-0.5, -0.5, -0.5],
            vec![-0.5, 0.5, -0.5],
            vec![0.5, 0.5, -0.5],
            vec![0.5, -0.5, -0.5],
        ];
    let points = convert_to_buffer(get_verts());
    let uvs = convert_uvs_to_buffer(get_uvs());
    let normals = convert_to_buffer(get_normals());
    let indicies = convert_indicies_to_buffer(get_indicies());

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        // Each array is an [x, y, z] coordinate in local space.
        // The camera coordinate space is right-handed x-right, y-up, z-back. This means "forward" is -Z.
        // Meshes always rotate around their local [0, 0, 0] when a rotation is applied to their Transform.
        // By centering our mesh around the origin, rotating the mesh preserves its center of mass.
        points,
    )
    // Set-up UV coordinates to point to the upper (V < 0.5), "dirt+grass" part of the texture.
    // Take a look at the custom image (assets/textures/array_texture.png)
    // so the UV coords will make more sense
    // Note: (0.0, 0.0) = Top-Left in UV mapping, (1.0, 1.0) = Bottom-Right in UV mapping

    // For meshes with flat shading, normals are orthogonal (pointing out) from the direction of
    // the surface.
    // Normals are required for correct lighting calculations.
    // Each array represents a normalized vector, which length should be equal to 1.0.

    .with_inserted_indices(Indices::U32(indicies)) 
        .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals, 
    )

    .with_inserted_attribute(
        Mesh::ATTRIBUTE_UV_0,
        uvs, 
    ) 
}

// Function that changes the UV mapping of the mesh, to apply the other texture.
fn toggle_texture(mesh_to_change: &mut Mesh) {
    // Get a mutable reference to the values of the UV attribute, so we can iterate over it.
    let uv_attribute = mesh_to_change.attribute_mut(Mesh::ATTRIBUTE_UV_0).unwrap();
    // The format of the UV coordinates should be Float32x2.
    let VertexAttributeValues::Float32x2(uv_attribute) = uv_attribute else {
        panic!("Unexpected vertex format, expected Float32x2.");
    };

    // Iterate over the UV coordinates, and change them as we want.
    for uv_coord in uv_attribute.iter_mut() {
        // If the UV coordinate points to the upper, "dirt+grass" part of the texture...
        if (uv_coord[1] + 0.5) < 1.0 {
            // ... point to the equivalent lower, "sand+water" part instead,
            uv_coord[1] += 0.5;
        } else {
            // else, point back to the upper, "dirt+grass" part.
            uv_coord[1] -= 0.5;
        }
    }
}

fn get_verts() -> PyResult<Vec<[f32; 3]>> {
    Python::attach(|py| {
        // Add your script directory to sys.path so Python can find it
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let new_path = path.downcast()?;
        new_path.insert(0, "D:/rust/bevy_usd/scripts")?;

        // Import the Python module (filename without .py)
        let module = PyModule::import(py, "get_verts")?;

        // Call the function
        let verts: Vec<Vec<f32>> = module.getattr("get_verts")?.call0()?.extract()?;
        let verts: Vec<[f32; 3]> = verts
            .into_iter()
            .map(|v| v.try_into().expect("Expected a Vec of length 3"))
            .collect();
        Ok(verts)
    })
}

fn get_uvs() -> PyResult<Vec<[f32; 2]>> {
    Python::attach(|py| {
        // Add your script directory to sys.path so Python can find it
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let new_path = path.downcast()?;
        new_path.insert(0, "D:/rust/bevy_usd/scripts")?;

        // Import the Python module (filename without .py)
        let module = PyModule::import(py, "get_verts")?;

        // Call the function
        let verts: Vec<Vec<f32>> = module.getattr("get_uvs")?.call0()?.extract()?;
        let verts: Vec<[f32; 2]> = verts
            .into_iter()
            .map(|v| v.try_into().expect("Expected a Vec of length 3"))
            .collect();
        Ok(verts)
    })
}

fn get_indicies() -> PyResult<Vec<u32>> {
    Python::attach(|py| {
        // Add your script directory to sys.path so Python can find it
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let new_path = path.downcast()?;
        new_path.insert(0, "D:/rust/bevy_usd/scripts")?;

        // Import the Python module (filename without .py)
        let module = PyModule::import(py, "get_verts")?;

        // Call the function
        let verts: Vec<u32> = module.getattr("get_indicies")?.call0()?.extract()?;
        Ok(verts)
    })
}

fn get_normals() -> PyResult<Vec<[f32; 3]>> {
    Python::attach(|py| {
        // Add your script directory to sys.path so Python can find it
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let new_path = path.downcast()?;
        new_path.insert(0, "D:/rust/bevy_usd/scripts")?;

        // Import the Python module (filename without .py)
        let module = PyModule::import(py, "get_verts")?;

        // Call the function
        let verts: Vec<Vec<f32>> = module.getattr("get_normals")?.call0()?.extract()?;
        let verts: Vec<[f32; 3]> = verts
            .into_iter()
            .map(|v| v.try_into().expect("Expected a Vec of length 3"))
            .collect();
        Ok(verts)
    })
}

fn convert_to_buffer(verts: PyResult<Vec<[f32; 3]>>) -> Vec<[f32; 3]> {
    match verts {
        Ok(points) => return points,
        Err(_) => return vec![[0.0, 0.0, 0.0]],
    };
}

fn convert_uvs_to_buffer(verts: PyResult<Vec<[f32; 2]>>) -> Vec<[f32; 2]> {
    match verts {
        Ok(points) => return points,
        Err(_) => return vec![[0.0, 0.0]],
    };
}

fn convert_indicies_to_buffer(verts: PyResult<Vec<u32>>) -> Vec<u32> {
    match verts {
        Ok(points) => return points,
        Err(_) => return vec![0],
    };
}
