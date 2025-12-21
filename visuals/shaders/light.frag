#version 430

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
in vec4 FragPosLightSpace;

out vec4 color;

uniform sampler2D ourTexture;
uniform vec3 viewPos;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float reflectivity;
}; 

struct DirectionLight
{
    int enable;
    vec3 direction;
    vec3 lightColor;
};

struct PointLight {
    int enable;
    vec3 position;  
    vec3 lightColor;

    float constant;
    float linear;
    float quadratic;
};

struct Spotlight {
    int enable;
    vec3 position;
    vec3 direction;
    vec3 lightColor;
    float cutOff;

    float constant;
    float linear;
    float quadratic;      
}; 

uniform Material material;
uniform DirectionLight dl;
uniform PointLight pl;
uniform Spotlight sl;
uniform samplerCube skybox;
// TODO#3-4: fragment shader
// Note:
//           1. how to write a fragment shader:
//              a. The output is FragColor (any var is OK)
//           2. Calculate
//              a. For direct light, lighting = ambient + diffuse + specular
//              b. For point light & spot light, lighting = ambient + attenuation * (diffuse + specular)
//              c. Final color = direct + point + spot if all three light are enabled
//           3. attenuation
//              a. Use formula from slides 'shading.ppt' page 20
//           4. For each light, ambient, diffuse and specular color are the same
//           5. diffuse = kd * max(normal vector dot light direction, 0.0)
//           6. specular = ks * pow(max(normal vector dot halfway direction), 0.0), shininess);
//           7. we've set all light parameters for you (see context.h) and definition in fragment shader
//           8. You should pre calculate normal matrix (trasfer normal from model local space to world space)
//              in light.cpp rather than in shaders
vec3 calculateDirectionLight() {
    if (dl.enable == 0) return vec3(0.0);
    vec3 albedo = texture(ourTexture, TexCoord).rgb;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-dl.direction);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 ambient = material.ambient * albedo * dl.lightColor;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = material.diffuse * diff * albedo * dl.lightColor;
    float spec = pow(max(dot(norm, halfwayDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec * dl.lightColor;
    return ambient + diffuse + specular;
}

vec3 calculatePointLight() {
    if (pl.enable == 0) return vec3(0.0);
    vec3 albedo = texture(ourTexture, TexCoord).rgb;
    vec3 norm = normalize(Normal);
    vec3 lightVector = pl.position - FragPos;
    vec3 lightDir = normalize(lightVector);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 ambient = material.ambient * albedo * pl.lightColor;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = material.diffuse * diff * albedo * pl.lightColor;
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec * pl.lightColor;
    float distance = length(lightVector);
    float attenuation = 1.0 / (pl.constant + pl.linear * distance + pl.quadratic * distance * distance);
    return ambient + attenuation * (diffuse + specular);
}

vec3 calculateSpotLight() {
    if (sl.enable == 0) return vec3(0.0);
    vec3 albedo = texture(ourTexture, TexCoord).rgb;
    vec3 norm = normalize(Normal);
    vec3 lightVector = sl.position - FragPos;
    vec3 lightDir = normalize(lightVector);
    vec3 spotDir = normalize(-sl.direction);
    float theta = dot(lightDir, spotDir);
    if (theta <= sl.cutOff) return vec3(0.0);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 ambient = material.ambient * albedo * sl.lightColor;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = material.diffuse * diff * albedo * sl.lightColor;
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), material.shininess);
    vec3 specular = material.specular * spec * sl.lightColor;
    float distance = length(lightVector);
    float attenuation = 1.0 / (sl.constant + sl.linear * distance + sl.quadratic * distance * distance);
    return ambient + attenuation * (diffuse + specular);
}

void main() {
    vec3 lighting = calculateDirectionLight() + calculatePointLight() + calculateSpotLight();
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflected = texture(skybox, reflect(-viewDir, norm)).rgb;
    float reflectivity = clamp(material.reflectivity, 0.0, 1.0);
    vec3 finalColor = mix(lighting, reflected, reflectivity);
    color = vec4(finalColor, 1.0);
}
