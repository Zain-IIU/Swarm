using UnityEngine;

namespace SwarmTest
{
    [CreateAssetMenu(menuName = "Swarm/Creature Profile")]
    public class CreatureProfile : ScriptableObject
    {
        [Header("Model")]
        public GameObject prefab;
        public string bonePrefix = "Spine_";
        public int boneCount = 10;
        [Tooltip("Collision radius per joint, head first, boneCount + 1 values (last = tail tip)")]
        public float[] jointRadius = new float[0];

        [Header("Locomotion")]
        public float moveSpeed = 0.55f;
        [Range(0f, 0.5f)] public float speedVariation = 0.25f;
        [Tooltip("Degrees per second")]
        public float turnRate = 160f;
        [Tooltip("Degrees")]
        public float wiggleAmplitude = 30f;
        public float wiggleFrequency = 1.4f;
        [Range(0f, 1f)] public float wanderStrength = 0.3f;
        public float climbStrength = 1.0f;
        [Tooltip("Climbing over others: lower = flatter swarm, higher = taller mounds")]
        public float pileClimbStrength = 0.35f;
        [Tooltip("Touching body segments push themselves along the path toward the head")]
        [Range(0f, 1.5f)] public float bodyThrust = 0.6f;

        [Header("Body")]
        [Range(0f, 1f)] public float lateralFriction = 0.5f;
        [Range(0f, 1f)] public float forwardFriction = 0.08f;
        [Tooltip("Friction while resting only on others; low keeps piles flowing")]
        [Range(0f, 1f)] public float pileFriction = 0.12f;
        [Range(0f, 1f)] public float bendStiffness = 0.08f;
        [Tooltip("Joint i and i+2 can't get closer than this fraction of their rest distance")]
        [Range(0.1f, 1f)] public float minFoldRatio = 0.55f;
        [Range(0f, 1f)] public float collisionStiffness = 0.8f;
    }
}
