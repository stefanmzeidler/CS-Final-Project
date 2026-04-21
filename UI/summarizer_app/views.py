from django.shortcuts import render
from django.http import HttpResponse
from .summarizer_service import get_summarizer
from .models import Summary


def summarize(request):
    pmc_id = request.GET.get("pmc_id")
    print(pmc_id)
    if pmc_id:
        try:
            article = Summary.objects.get(pmc_id=pmc_id)
            print("Found existing summaries in database")
            # summaries = {
            #     "basic_summary": article.basic_summary,
            #     "college_summary": article.college_summary,
            #     "professional_summary": article.professional_summary,
            # }
            return render(
                request,
                template_name="summarizer_app/summarizer.html",
                context={"basic_summary":article.basic_summary, "college_summary":article.college_summary, "professional_summary": article.professional_summary},
            )
        except Summary.DoesNotExist:
            print("No existing summaries in database found")
            print("Generating summaries for pmcd id: {}".format(pmc_id))
            summarizer = get_summarizer()
            summaries = summarizer.summarize(pmc_id)
            basic_summary=summaries["basic"]
            college_summary = summaries["college"]
            professional_summary = (summaries["professional"],)
            new_summaries = Summary(
                pmc_id=pmc_id,
                basic_summary=basic_summary,
                college_summary=college_summary,
                professional_summary=professional_summary,
            )
            new_summaries.save()
            print("Summaries generated.")
            return render(
                request,
                template_name="summarizer_app/summarizer.html",
                context={"basic_summary": basic_summary,"college_summary": college_summary, "professional_summary": professional_summary},
            )
    else:
        return render(
            request,
            template_name="summarizer_app/summarizer.html",
        )
