use bevy::{
    prelude::*,
    input::mouse::MouseMotion,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
};
use pyo3::prelude::*;
use pyo3::types::{PyModule,PyString,PyAny};

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
        .add_systems(Update, camera)
        .add_systems(Update, rotate_camera_to_mouse)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (points, normals, uvs, indicies, xform) = get_data().unwrap();
    let custom_texture_handle: Handle<Image> = asset_server.load("textures/test.png");
    
    
    for (index, point) in points.iter().enumerate(){
        let vector = xform[index].clone();
        let matrix = vec_to_mat4(vector);
        let point_attr = point.clone();
        let normal_attr = normals[index].clone();
        let uv_attr = uvs[index].clone();
        let indicies_attr = indicies[index].clone();
        
        let cube_mesh_handle: Handle<Mesh> = meshes.add(create_cube_mesh(point_attr, normal_attr, uv_attr, indicies_attr));
            // Render the mesh with the custom texture, and add the marker.
        
            commands.spawn((
                Mesh3d(cube_mesh_handle),
                Transform::from_matrix(matrix),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color_texture: Some(custom_texture_handle.clone()),
                    ..default()
                })),
                CustomUV,
            ));

    }

    // Import the custom texture.

    // Create and save a handle to the mesh.
    
    
   

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


fn camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query:  Query<&mut Transform, With<Camera>>,
    time: Res<Time>,
) {
    if keyboard_input.pressed(KeyCode::KeyW) {
        for mut cam in &mut query{
            cam.translation.z-=1.0*time.delta_secs();
        }
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        for mut cam in &mut query{
            cam.translation.z+=1.0*time.delta_secs();
        }
    }

    if keyboard_input.pressed(KeyCode::KeyA) {
        for mut cam in &mut query{
            cam.translation.x-=1.0*time.delta_secs();
        }
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        for mut cam in &mut query{
            cam.translation.x+=1.0*time.delta_secs();
        }
    }
}


fn rotate_camera_to_mouse(
  time: Res<Time>,
  mut mouse_motion: EventReader<MouseMotion>,
  mut transform: Single<&mut Transform, With<Camera>>,
) {
  let dt = time.delta_secs();
  // The factors are just arbitrary mouse sensitivity values.
  // It's often nicer to have a faster horizontal sensitivity than vertical.
  let mouse_sensitivity = Vec2::new(0.12, 0.10);

  for motion in mouse_motion.read() {
    let delta_yaw = -motion.delta.x * dt * mouse_sensitivity.x;
    let delta_pitch = -motion.delta.y * dt * mouse_sensitivity.y;

    // Add yaw which is turning left/right (global)
    transform.rotate_y(delta_yaw);

    // Add pitch which is looking up/down (local)
    const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
    let pitch = (pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);

    // Apply the rotation
    transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
  }
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
fn create_cube_mesh(points: Vec<[f32;3]>, normals: Vec<[f32;3]>, uvs: Vec<[f32;2]>, indicies:Vec<u32>) -> Mesh {
    // Keep the mesh data accessible in future frames to be able to mutate it in toggle_texture.
    //println!("{:?}", get_verts());

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

#[pyfunction]
fn get_data() ->  PyResult<(Vec<Vec<[f32;3]>>,Vec<Vec<[f32;3]>>,Vec<Vec<[f32;2]>>,Vec<Vec<u32>>, Vec<Vec<[f32;4]>>)> {
    Python::attach(|py| {
        // Add your script directory to sys.path so Python can find it
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let new_path = path.downcast()?;
        new_path.insert(0, "C:/development/rust/bevy_usd/scripts")?;

        // Import the Python module (filename without .py)
        let module = PyModule::import(py, "get_data")?;
        let usd_file = PyString::new(py, "C:/development/rust/bevy_usd/scripts/test.py");
        let data= module.getattr("get_data")?.call1((usd_file,))?;
        let points:Vec<Vec<[f32;3]>> = data.call_method0("get_points")?.extract()?;
        let normals:Vec<Vec<[f32;3]>> = data.call_method0("get_normals")?.extract()?;
        let uvs:Vec<Vec<[f32;2]>> = data.call_method0("get_uvs")?.extract()?;
        let indicies:Vec<Vec<u32>> = data.call_method0("get_indicies")?.extract()?;
        let xform: Vec<Vec<[f32;4]>>= data.call_method0("get_matrix")?.extract()?;
        let result = (points, normals, uvs, indicies, xform);
        Ok(result)
    })
}


fn vec_to_mat4(matrix: Vec<[f32; 4]>) -> Mat4 {
    Mat4 {
        x_axis: vec4(matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]),
        y_axis: vec4(matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]),
        z_axis: vec4(matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]),
        w_axis: vec4(matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]),
    }
}

fn extract_translation_and_scale(matrix: &Vec<[f32; 4]>) -> ([f32; 3], [f32; 3]) {
    // Translation (slate)
    let translation = [matrix[0][3], matrix[1][3], matrix[2][3]];

    // Helper function to compute length of first 3 elements in a row
    fn vec3_length(row: &[f32; 4]) -> f32 {
        (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt()
    }

    // Scale for each axis
    let scale_x = vec3_length(&matrix[0]);
    let scale_y = vec3_length(&matrix[1]);
    let scale_z = vec3_length(&matrix[2]);

    let scale = [scale_x, scale_y, scale_z];

    (translation, scale)
}