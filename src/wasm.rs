#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    use crate::api::types::*;

    #[wasm_bindgen]
    pub fn create_chat_info(id: &str, title: &str, user_excerpt: &str, created_at: &str, updated_at: &str) -> JsValue {
        let chat_info = ChatInfo {
            id: id.to_string(),
            title: title.to_string(),
            user_excerpt: user_excerpt.to_string(),
            created_at: created_at.to_string(),
            updated_at: updated_at.to_string(),
        };
        serde_wasm_bindgen::to_value(&chat_info).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_chats_response(chat_infos: JsValue) -> JsValue {
        let chat_infos: Vec<ChatInfo> = serde_wasm_bindgen::from_value(chat_infos).unwrap_or_default();
        let response = ChatsResponse { chat_infos };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_chat(id: &str, title: &str) -> JsValue {
        let chat = Chat {
            id: id.to_string(),
            messages: Vec::new(),
            title: title.to_string(),
            created_at: None,
            browser_state: None,
        };
        serde_wasm_bindgen::to_value(&chat).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_message(role: &str, content: &str) -> JsValue {
        let message = Message {
            role: role.to_string(),
            content: content.to_string(),
            thinking: None,
            stream: None,
            model: None,
            attachments: None,
            tool_calls: None,
            tool_call: None,
            tool_name: None,
            tool_result: None,
            created_at: None,
            updated_at: None,
            thinking_time_start: None,
            thinking_time_end: None,
        };
        serde_wasm_bindgen::to_value(&message).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_chat_response(chat: JsValue) -> JsValue {
        let chat: Chat = serde_wasm_bindgen::from_value(chat).unwrap_or(Chat {
            id: String::new(),
            messages: Vec::new(),
            title: String::new(),
            created_at: None,
            browser_state: None,
        });
        let response = ChatResponse { chat };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_model(model_name: &str) -> JsValue {
        let model = Model::new(model_name);
        serde_wasm_bindgen::to_value(&model).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_models_response(models: JsValue) -> JsValue {
        let models: Vec<Model> = serde_wasm_bindgen::from_value(models).unwrap_or_default();
        let response = ModelsResponse { models };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_inference_compute(library: &str, variant: &str, compute: &str, driver: &str, name: &str, vram: &str) -> JsValue {
        let compute = InferenceCompute {
            library: library.to_string(),
            variant: variant.to_string(),
            compute: compute.to_string(),
            driver: driver.to_string(),
            name: name.to_string(),
            vram: vram.to_string(),
        };
        serde_wasm_bindgen::to_value(&compute).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_inference_compute_response(inference_computes: JsValue, default_context_length: u32) -> JsValue {
        let inference_computes: Vec<InferenceCompute> = serde_wasm_bindgen::from_value(inference_computes).unwrap_or_default();
        let response = InferenceComputeResponse {
            inference_computes,
            default_context_length: Some(default_context_length),
        };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_model_capabilities_response(capabilities: JsValue) -> JsValue {
        let capabilities: Vec<String> = serde_wasm_bindgen::from_value(capabilities).unwrap_or_default();
        let response = ModelCapabilitiesResponse::with_capabilities(capabilities);
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_chat_event(event_name: &str, content: &str) -> JsValue {
        let event_name = match event_name {
            "chat" => ChatEventName::Chat,
            "thinking" => ChatEventName::Thinking,
            "assistant_with_tools" => ChatEventName::AssistantWithTools,
            "tool_call" => ChatEventName::ToolCall,
            "tool" => ChatEventName::Tool,
            "tool_result" => ChatEventName::ToolResult,
            "done" => ChatEventName::Done,
            "chat_created" => ChatEventName::ChatCreated,
            _ => ChatEventName::Chat,
        };
        let event = ChatEvent {
            event_name,
            content: Some(content.to_string()),
            thinking: None,
            thinking_time_start: None,
            thinking_time_end: None,
            tool_calls: None,
            tool_call: None,
            tool_name: None,
            tool_result: None,
            tool_result_data: None,
            chat_id: None,
            tool_state: None,
        };
        serde_wasm_bindgen::to_value(&event).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_download_event(total: u64, completed: u64, done: bool) -> JsValue {
        let event = DownloadEvent {
            event_name: "download".to_string(),
            total,
            completed,
            done,
        };
        serde_wasm_bindgen::to_value(&event).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_error_event(error: &str, code: Option<String>, details: Option<String>) -> JsValue {
        let event = ErrorEvent {
            error: error.to_string(),
            event_name: "error".to_string(),
            code,
            details,
        };
        serde_wasm_bindgen::to_value(&event).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_settings() -> JsValue {
        let settings = Settings::new();
        serde_wasm_bindgen::to_value(&settings).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_settings_response(settings: JsValue) -> JsValue {
        let settings: Settings = serde_wasm_bindgen::from_value(settings).unwrap_or(Settings::new());
        let response = SettingsResponse { settings };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_health_response(healthy: bool) -> JsValue {
        let response = HealthResponse::new(healthy);
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_user(id: &str, email: &str, name: &str) -> JsValue {
        let user = User {
            id: id.to_string(),
            email: email.to_string(),
            name: name.to_string(),
            bio: None,
            avatar_url: None,
            firstname: None,
            lastname: None,
            plan: None,
        };
        serde_wasm_bindgen::to_value(&user).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_chat_request(model: &str, prompt: &str) -> JsValue {
        let request = ChatRequest::new(model, prompt);
        serde_wasm_bindgen::to_value(&request).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_error(error: &str) -> JsValue {
        let err = Error::new(error);
        serde_wasm_bindgen::to_value(&err).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_model_upstream_response(digest: Option<String>, push_time: u64, error: Option<String>) -> JsValue {
        let response = ModelUpstreamResponse {
            digest,
            push_time,
            error,
        };
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_cloud_status_response(disabled: bool, source: &str) -> JsValue {
        let response = CloudStatusResponse::new(disabled, source);
        serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_page(url: &str, title: &str, text: &str) -> JsValue {
        let page = Page {
            url: url.to_string(),
            title: title.to_string(),
            text: text.to_string(),
            lines: Vec::new(),
            links: None,
            fetched_at: None,
        };
        serde_wasm_bindgen::to_value(&page).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_browser_state_data() -> JsValue {
        let state = BrowserStateData {
            page_stack: Vec::new(),
            view_tokens: None,
            url_to_page: None,
        };
        serde_wasm_bindgen::to_value(&state).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_tool_function(name: &str, arguments: &str) -> JsValue {
        let func = ToolFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
            result: None,
        };
        serde_wasm_bindgen::to_value(&func).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_tool_call(call_type: &str, function: JsValue) -> JsValue {
        let function: ToolFunction = serde_wasm_bindgen::from_value(function).unwrap_or(ToolFunction {
            name: String::new(),
            arguments: String::new(),
            result: None,
        });
        let tool_call = ToolCall {
            call_type: call_type.to_string(),
            function,
        };
        serde_wasm_bindgen::to_value(&tool_call).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_file(filename: &str, data: JsValue) -> JsValue {
        let data: Vec<u8> = serde_wasm_bindgen::from_value(data).unwrap_or_default();
        let file = File {
            filename: filename.to_string(),
            data,
        };
        serde_wasm_bindgen::to_value(&file).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_attachment(filename: &str, data: Option<String>) -> JsValue {
        let attachment = Attachment {
            filename: filename.to_string(),
            data,
        };
        serde_wasm_bindgen::to_value(&attachment).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn serialize_to_json(value: JsValue) -> String {
        serde_json::to_string(&value).unwrap_or_default()
    }

    #[wasm_bindgen]
    pub fn deserialize_from_json(json: &str) -> JsValue {
        serde_json::from_str(json).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_settings_state(
        turbo_enabled: bool,
        web_search_enabled: bool,
        selected_model: &str,
        sidebar_open: bool,
        think_enabled: bool,
        think_level: &str,
    ) -> JsValue {
        let state = SettingsState {
            turbo_enabled,
            web_search_enabled,
            selected_model: selected_model.to_string(),
            sidebar_open,
            think_enabled,
            think_level: think_level.to_string(),
        };
        serde_wasm_bindgen::to_value(&state).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_settings_state_from_settings(settings: JsValue) -> JsValue {
        let settings: Settings = serde_wasm_bindgen::from_value(settings).unwrap_or(Settings::new());
        let state = SettingsState::from_settings(&settings);
        serde_wasm_bindgen::to_value(&state).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_batcher_config(batch_interval: u32, immediate_first: bool) -> JsValue {
        let config = BatcherConfig {
            batch_interval: Some(batch_interval),
            immediate_first: Some(immediate_first),
        };
        serde_wasm_bindgen::to_value(&config).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_image_data(filename: &str, path: &str, data_url: &str) -> JsValue {
        let image_data = ImageData::new(filename, path, data_url);
        serde_wasm_bindgen::to_value(&image_data).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_menu_item(label: &str, enabled: Option<bool>, separator: Option<bool>) -> JsValue {
        let item = MenuItem {
            label: label.to_string(),
            enabled,
            separator,
        };
        serde_wasm_bindgen::to_value(&item).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_use_message_autoscroll_options(messages: JsValue, is_streaming: bool, chat_id: &str) -> JsValue {
        let messages: Vec<Message> = serde_wasm_bindgen::from_value(messages).unwrap_or_default();
        let options = UseMessageAutoscrollOptions {
            messages,
            is_streaming,
            chat_id: chat_id.to_string(),
        };
        serde_wasm_bindgen::to_value(&options).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn create_message_autoscroll_behavior(spacer_height: u32) -> JsValue {
        let behavior = MessageAutoscrollBehavior {
            spacer_height,
        };
        serde_wasm_bindgen::to_value(&behavior).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn uint8_array_to_base64(data: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(data)
    }

    #[wasm_bindgen]
    pub fn base64_to_uint8_array(base64_str: &str) -> Vec<u8> {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(base64_str).unwrap_or_default()
    }

    #[wasm_bindgen]
    pub fn model_is_cloud(model: &str) -> bool {
        model.ends_with("cloud")
    }
}
