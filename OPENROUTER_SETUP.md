# إعداد OpenRouter للتطبيق

## الحصول على مفتاح OpenRouter

1. **التسجيل في OpenRouter:**
   - اذهب إلى: https://openrouter.ai/
   - قم بإنشاء حساب جديد أو تسجيل الدخول

2. **إنشاء مفتاح API:**
   - اذهب إلى صفحة API Keys: https://openrouter.ai/keys
   - انقر على "Create Key"
   - انسخ المفتاح واحتفظ به بأمان

## إعداد ملف .env

قم بتحديث ملف `.env` بمفتاح OpenRouter الخاص بك:

```properties
# ضع مفتاح OpenRouter هنا
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# النماذج المتاحة في OpenRouter
LLM_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# عناوين API
LLM_API_BASE=https://openrouter.ai/api/v1
EMBEDDING_API_BASE=https://openrouter.ai/api/v1
```

## النماذج المتاحة

### نماذج اللغة (LLM):
- `openai/gpt-4o-mini` - سريع ورخيص
- `openai/gpt-4o` - قوي ودقيق
- `anthropic/claude-3-haiku` - بديل سريع
- `anthropic/claude-3-sonnet` - متوازن
- `meta-llama/llama-3-8b-instruct` - مجاني ومفتوح المصدر

### نماذج التضمين (Embeddings):
- `text-embedding-3-small` - OpenAI (موصى به)
- `text-embedding-3-large` - OpenAI (دقة أعلى)

## مزايا OpenRouter

1. **تنوع النماذج:** الوصول لنماذج متعددة من مقدمين مختلفين
2. **التكلفة:** أسعار تنافسية وشفافة
3. **المرونة:** تغيير النماذج بسهولة دون تغيير الكود
4. **الموثوقية:** خدمة مستقرة مع إحصائيات مفصلة

## الاستخدام

بعد إعداد المفتاح، شغل التطبيق كالمعتاد:

```bash
python run_cli.py run ../data/examples/sample_research.pdf
```

## استكشاف الأخطاء

- **خطأ 401:** تأكد من صحة مفتاح API
- **خطأ 429:** تم تجاوز حدود الاستخدام
- **خطأ الاتصال:** تحقق من الاتصال بالإنترنت

## النسخ الاحتياطي

في حالة عدم توفر مفتاح صالح، سيتحول التطبيق تلقائيًا إلى:
1. نماذج HuggingFace المحلية للتضمين
2. استجابات وهمية للتحليل (للاختبار فقط)
