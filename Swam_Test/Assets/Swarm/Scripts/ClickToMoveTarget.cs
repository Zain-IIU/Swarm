using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace SwarmTest
{
    public class ClickToMoveTarget : MonoBehaviour
    {
        public Camera cam;
        public LayerMask mask = ~0;
        public float heightOffset = 0.05f;
        [Tooltip("0 = left, 1 = right, 2 = middle")]
        public int mouseButton = 0;

        [Header("Repeller controls (if this object has a SwarmRepeller)")]
        public float scrollRadiusStep = 0.15f;
        public Vector2 radiusLimits = new Vector2(0.3f, 4f);
        [Tooltip("Key that switches the linked light on and off")]
        public KeyCode toggleKey = KeyCode.F;

        SwarmRepeller repeller;

        void Start() { repeller = GetComponent<SwarmRepeller>(); }

        void Update()
        {
            if (cam == null) cam = Camera.main;
            if (cam == null) return;

            bool held, toggle;
            float scroll;
            Vector2 mouse;
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current == null) return;
            var m = Mouse.current;
            held = (mouseButton == 1 ? m.rightButton : mouseButton == 2 ? m.middleButton : m.leftButton).isPressed;
            mouse = m.position.ReadValue();
            scroll = m.scroll.ReadValue().y / 120f;
            toggle = Keyboard.current != null && Keyboard.current[ToKey(toggleKey)].wasPressedThisFrame;
#else
            held = Input.GetMouseButton(mouseButton);
            mouse = Input.mousePosition;
            scroll = Input.mouseScrollDelta.y;
            toggle = Input.GetKeyDown(toggleKey);
#endif
            if (repeller != null)
            {
                if (Mathf.Abs(scroll) > 0.01f)
                    repeller.radius = Mathf.Clamp(repeller.radius + scroll * scrollRadiusStep, radiusLimits.x, radiusLimits.y);
                if (toggle && repeller.linkedLight != null)
                    repeller.linkedLight.enabled = !repeller.linkedLight.enabled;
            }

            if (!held) return;
            RaycastHit hit;
            if (Physics.Raycast(cam.ScreenPointToRay(mouse), out hit, 500f, mask, QueryTriggerInteraction.Ignore))
                transform.position = hit.point + hit.normal * heightOffset;
        }

#if ENABLE_INPUT_SYSTEM
        static Key ToKey(KeyCode k)
        {
            if (k >= KeyCode.A && k <= KeyCode.Z) return Key.A + (k - KeyCode.A);
            if (k == KeyCode.Space) return Key.Space;
            return Key.F;
        }
#endif
    }
}
